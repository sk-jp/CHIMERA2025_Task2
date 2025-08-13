import os
import csv
import cv2
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import openslide
from gigapath.slide_encoder import create_model as create_slide_encoder
import json
import random
import sys
torch.multiprocessing.set_sharing_strategy('file_system')

class WSITileDataset(Dataset):
    """
    Tile dataset for WSI

    Arguments:
    ----------
    wsi_path : str
        Path to WSI image
    mask_path : str
        Path to mask image
    level : int
        Level to use (corresponds to 0.5um/pixel)
    tile_size : int
        Tile size (224x224)
    transform : transforms.Compose
        Preprocessing transforms
    """
    
    def __init__(self, wsi_path: str, mask_path: str, level: int, tile_size: int = 224, transform=None,
                 num_max_tiles: int = 50000):
        self.wsi_path = wsi_path
        self.mask_path = mask_path
        self.level = level
        self.tile_size = tile_size
        self.transform = transform
        
        # Load WSI and mask
        self.wsi = openslide.OpenSlide(wsi_path)
        import tifffile
        self.mask = tifffile.imread(mask_path)
        if self.mask.ndim == 3:
            # If mask is 3 channels, convert to grayscale
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_RGB2GRAY)
        
        # Generate list of valid tile coordinates
        self.valid_tiles = self._generate_valid_tiles(num_max_tiles)
        
    def _generate_valid_tiles(self, num_max_tiles) -> List[Tuple[int, int]]:
        """
        Generate valid tile coordinates based on mask
        """
        valid_tiles = []
        
        # Get image size at level 0
        level_0_width, level_0_height = self.wsi.level_dimensions[0]
        
        # Get image size at current level
        level_width, level_height = self.wsi.level_dimensions[self.level]
        
        # Calculate downsampling ratio
        downsample = self.wsi.level_downsamples[self.level]
        
        # Get mask size
        mask_height, mask_width = self.mask.shape
        
        # Convert tile size to level 0 coordinates
        tile_size_level0 = int(self.tile_size * downsample)
        
        # Process in y, x order (to match OpenCV coordinate system)
        for y in range(0, level_0_height - tile_size_level0, tile_size_level0):
            for x in range(0, level_0_width - tile_size_level0, tile_size_level0):
                # Convert to mask coordinates (mask is not resized)
                mask_x = int(x * mask_width / level_0_width)
                mask_y = int(y * mask_height / level_0_height)
                mask_tile_w = int(tile_size_level0 * mask_width / level_0_width)
                mask_tile_h = int(tile_size_level0 * mask_height / level_0_height)
                
                # Check mask range
                if (mask_x + mask_tile_w <= mask_width and 
                    mask_y + mask_tile_h <= mask_height):
                    
                    # Get corresponding region from mask
                    mask_region = self.mask[mask_y:mask_y+mask_tile_h, mask_x:mask_x+mask_tile_w]
                    
                    # Calculate ratio of valid pixels
                    valid_pixel_ratio = np.mean(mask_region > 0)
                    
                    # Add as valid tile if above threshold
                    if valid_pixel_ratio > 0.1:  # More than 10% valid region
                        valid_tiles.append((x, y))

        # If too many tiles, randomly sample
        print(f" Number of valid tiles: {len(valid_tiles)}")
        if len(valid_tiles) > num_max_tiles:
            print(f" Sampling {num_max_tiles} valid tiles")
            random.seed(42)
            valid_tiles = random.sample(valid_tiles, num_max_tiles)

        return valid_tiles
    
    def __len__(self):
        return len(self.valid_tiles)
    
    def __getitem__(self, idx):
        x, y = self.valid_tiles[idx]
        
        # Read tile from WSI (OpenSlide uses (x, y) order)
        tile = self.wsi.read_region((x, y), self.level, (self.tile_size, self.tile_size))
        tile = tile.convert('RGB')
        
        # Convert from PIL Image to NumPy array
        if self.transform:
            tile = self.transform(tile)
        
        # Normalize coordinates by tile size (NumPy/PIL uses (x, y) order)
        coords = np.array([x, y], dtype=np.float32)
        
        return {
            'image': tile,
            'coords': torch.from_numpy(coords)
        }


def find_level_for_target_mpp(slide_path: str, target_mpp: float = 0.5) -> int:
    """
    Find the level closest to the specified MPP

    Arguments:
    ----------
    slide_path : str
        Path to slide file
    target_mpp : float
        Target MPP (default: 0.5um/pixel)

    Returns:
    --------
    int: Level closest to target MPP
    """
    slide = openslide.OpenSlide(slide_path)
    
    # Get resolution info from properties
    try:
        x_resolution = float(slide.properties.get('tiff.XResolution', 0))
        y_resolution = float(slide.properties.get('tiff.YResolution', 0))
        resolution_unit = slide.properties.get('tiff.ResolutionUnit', '')
        
        # Convert resolution to MPP
        if resolution_unit == 'centimeter' and x_resolution > 0:
            mpp_x = 10000 / x_resolution
            mpp_y = 10000 / y_resolution
        else:
            # Use default value
            print(f" Resolution information not available for {slide_path}. Using level 0.")
            return 0
        
        # Calculate MPP for each level and find the closest
        best_level = 0
        best_diff = float('inf')
        
        for level in range(slide.level_count):
            level_mpp_x = mpp_x * slide.level_downsamples[level]
            level_mpp_y = mpp_y * slide.level_downsamples[level]
            avg_mpp = (level_mpp_x + level_mpp_y) / 2
            
            diff = abs(avg_mpp - target_mpp)
            if diff < best_diff:
                best_diff = diff
                best_level = level
        
        print(f" Level {best_level} selected for {slide_path} (MPP: {avg_mpp:.3f})")
        return best_level
    
    except Exception as e:
        print(f" Error processing {slide_path}: {e}. Using level 0.")
        return 0


def load_tile_encoder_transforms(tile_size=224) -> transforms.Compose:
    """Load preprocessing transforms for tile encoder"""
    transform = transforms.Compose([
#        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
#        transforms.CenterCrop(tile_size),
#        transforms.RandomCrop(tile_size, padding=None),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform


def load_models(local_tile_encoder_path: str = '', 
                local_slide_encoder_path: str = '', 
                global_pool: bool = False) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load Prov-GigaPath tile encoder and slide encoder

    Arguments:
    ----------
    local_tile_encoder_path : str
        Local tile encoder path
    local_slide_encoder_path : str
        Local slide encoder path
    global_pool : bool
        Whether to use global pooling

    Returns:
    --------
    Tuple[torch.nn.Module, torch.nn.Module]: tile encoder and slide encoder
    """
    pretrain_dir = "/data/pretrained_models/Prov_GigaPath"

    # Initialize model
    print(" Loading tile encoder...")
    config = json.load(open(os.path.join(pretrain_dir, "config.json")))
    tile_encoder = timm.create_model(
        model_name=config['architecture'],
        checkpoint_path=os.path.join(pretrain_dir, "pytorch_model.bin"),
        **config["model_args"]
    )

    print(" Tile encoder param #", sum(p.numel() for p in tile_encoder.parameters()))
    
    # Loading pretrained models locally
    print(f" Loading slide encoder with global_pool={global_pool}...")
    slide_encoder_model = create_slide_encoder(
        pretrained=os.path.join(pretrain_dir, "slide_encoder.pth"),
        model_arch="gigapath_slide_enc12l768d",
        in_chans=1536,
        global_pool=global_pool
    )

    print(" Slide encoder param #", sum(p.numel() for p in slide_encoder_model.parameters()))
    
    return tile_encoder, slide_encoder_model


def normalize_coordinates(coords: torch.Tensor, tile_size: int = 224) -> torch.Tensor:
    """
    Normalize coordinates to the format expected by the slide encoder

    Arguments:
    ----------
    coords : torch.Tensor
        Original coordinates (N, 2)
    tile_size : int
        Tile size (default: 256)

    Returns:
    --------
    torch.Tensor: Normalized coordinates
    """
    # Normalize coordinates by dividing by tile size
    normalized_coords = coords.float() / tile_size
    
    # Convert coordinates to integer grid indices
    grid_coords = torch.floor(normalized_coords).long()
    
    return grid_coords


def clip_coordinates_to_grid(coords: torch.Tensor, max_grid_size: int = 1000) -> torch.Tensor:
    """
    Clip coordinates within the maximum grid size of the slide encoder

    Arguments:
    ----------
    coords : torch.Tensor
        Grid coordinates (N, 2)
    max_grid_size : int
        Maximum grid size (default: 1000)

    Returns:
    --------
    torch.Tensor: Clipped coordinates
    """
    # Clip coordinates to the range 0 to max_grid_size-1
    clipped_coords = torch.clamp(coords, 0, max_grid_size - 1)
    
    return clipped_coords


@torch.no_grad()
def extract_tile_features(dataset: WSITileDataset, 
                         tile_encoder: torch.nn.Module, 
                         batch_size: int = 32,
                         device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Extract tile-level features using the tile encoder

    Arguments:
    ----------
    dataset : WSITileDataset
        WSI tile dataset
    tile_encoder : torch.nn.Module
        Tile encoder
    batch_size : int
        Batch size
    device : str
        Device to use

    Returns:
    --------
    Dict[str, torch.Tensor]: tile features and coordinates
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()
    
    all_features = []
    all_coords = []
    
    for batch in tqdm(dataloader, desc=' Extracting tile features'):
        images = batch['image'].to(device)
        coords = batch['coords']
        
        with torch.amp.autocast(dtype=torch.float16, device_type='cuda'):
            features = tile_encoder(images)
        
        all_features.append(features.cpu())
        all_coords.append(coords)
    
    return {
        'tile_features': torch.cat(all_features, dim=0),
        'coords': torch.cat(all_coords, dim=0)
    }


@torch.no_grad()
def extract_slide_features(tile_features: torch.Tensor,
                           coords: torch.Tensor,
                           slide_encoder: torch.nn.Module,
                           tile_size: int = 224,
                           device: str = 'cuda') -> torch.Tensor:
    """
    Extract slide-level features using the slide encoder

    Arguments:
    ----------
    tile_features : torch.Tensor
        Tile-level features
    coords : torch.Tensor
        Tile coordinates
    slide_encoder : torch.nn.Module
        Slide encoder
    device : str
        Device to use

    Returns:
    --------
    torch.Tensor: Slide-level features
    """

    print("Extracting slide features...")

    slide_encoder = slide_encoder.to(device)
    slide_encoder.eval()
    
    # Normalize coordinates (according to slide encoder's tile size)
    normalized_coords = normalize_coordinates(coords, tile_size=tile_size)
    
    # Clip coordinates within slide encoder's grid size
    clipped_coords = clip_coordinates_to_grid(normalized_coords, max_grid_size=1000)

    # Check coordinate range
    print(f" Coordinate range: x=[{clipped_coords[:, 0].min()}, {clipped_coords[:, 0].max()}], "
          f"y=[{clipped_coords[:, 1].min()}, {clipped_coords[:, 1].max()}]")
    
    # Add batch dimension
    tile_features = tile_features.unsqueeze(0).to(device)
    clipped_coords = clipped_coords.unsqueeze(0).to(device)

    try:
        with torch.amp.autocast(dtype=torch.float16, device_type='cuda'):
            slide_features = slide_encoder(tile_features, clipped_coords)
    except RuntimeError as e:
        print(f"Error in slide encoder: {e}")
        print(f" Tile features shape: {tile_features.shape}")
        print(f" Coordinates shape: {clipped_coords.shape}")
        print(f" Unique coordinates: {torch.unique(clipped_coords.view(-1, 2), dim=0).shape}")
        raise e
    
    print("Extracted slide features")

    return slide_features[0].squeeze(0).cpu()


def process_csv_file(top_dir: str,
                     csv_path: str, 
                     output_dir: str,
                     tile_encoder: torch.nn.Module,
                     slide_encoder: torch.nn.Module,
                     target_mpp: float = 0.5,
                     tile_size: int = 224,
                     batch_size: int = 32,
                     num_max_tiles: int = 50000,
                     device: str = 'cuda'):
    """
    Process WSI files listed in the CSV file and extract slide-level features

    Arguments:
    ----------
    top_dir : str
        Top directory of the dataset
    csv_path : str
        Path to input CSV file
    output_dir : str
        Output directory
    tile_encoder : torch.nn.Module
        Tile encoder
    slide_encoder : torch.nn.Module
        Slide encoder
    target_mpp : float
        Target MPP
    tile_size : int
        Tile size (224x224)
    batch_size : int
        Batch size
    device : str
        Device to use
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        wsi_mask_pairs = list(reader)
    
    # Prepare preprocessing transforms
    transform = load_tile_encoder_transforms(tile_size=tile_size)
    
    # Create error log file
    error_log_path = os.path.join(output_dir, "error_log.txt")
    
    # Process each WSI file
    for i, (wsi_path, mask_path) in enumerate(wsi_mask_pairs):
        try:
            # Join paths with top directory
            wsi_path = os.path.join(top_dir, wsi_path)
            mask_path = os.path.join(top_dir, mask_path)

            print(f"Processing {wsi_path} with mask {mask_path}")
            
            # Check if files exist
            if not os.path.exists(wsi_path):
                error_msg = f"WSI file not found: {wsi_path}"
                print(error_msg)
                with open(error_log_path, 'a') as f:
                    f.write(error_msg + "\n")
                continue
                
            if not os.path.exists(mask_path):
                error_msg = f"Mask file not found: {mask_path}"
                print(error_msg)
                with open(error_log_path, 'a') as f:
                    f.write(error_msg + "\n")
                continue
            
            # Find level corresponding to 0.5um/pixel
            level = find_level_for_target_mpp(wsi_path, target_mpp=target_mpp)
            
            # Create dataset
            dataset = WSITileDataset(wsi_path, mask_path, level, tile_size=tile_size, transform=transform,
                                     num_max_tiles=num_max_tiles)
            
            if len(dataset) == 0:
                error_msg = f"No valid tiles found for {wsi_path}"
                print(error_msg)
                with open(error_log_path, 'a') as f:
                    f.write(error_msg + "\n")
                continue
            
            print(f" Found {len(dataset)} valid tiles")
            
            # Extract tile-level features
            tile_results = extract_tile_features(dataset, tile_encoder, batch_size, device)
            
            # Extract slide-level features
            slide_features = extract_slide_features(
                tile_results['tile_features'],
                tile_results['coords'],
                slide_encoder,
                tile_size,
                device
            )
            
            # Save results
            wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
            output_path = os.path.join(output_dir, f"{wsi_name}.pt")
            
            """
            torch.save({
                'slide_features': slide_features,
                'tile_features': tile_results['tile_features'],
                'coords': tile_results['coords'],
                'wsi_path': wsi_path,
                'mask_path': mask_path,
                'level': level,
                'num_tiles': len(dataset)
            }, output_path)
            """
            torch.save(slide_features, output_path)
            
            print(f" Saved features for {wsi_name} to {output_path}")
            
            # Clear memory
            del dataset, tile_results, slide_features
            torch.cuda.empty_cache()
            
        except Exception as e:
            error_msg = f"Error processing {wsi_path}: {e}"
            print(error_msg)
            with open(error_log_path, 'a') as f:
                f.write(error_msg + "\n")
            
            # Clear memory
            torch.cuda.empty_cache()
            continue


def main(csv_path: str, gpu_id: int):
    """
    Main process
    """
    # Parameter settings
    top_dir = "/data/MICCAI2025_CHIMERA/task2/data"  # Top directory of the dataset
#    csv_path = "datalist.csv"  # Input CSV file
    output_dir = "/data/MICCAI2025_CHIMERA/task2/gigapath_slide_features_50000tiles"    # Output directory
    batch_size = 512               # Batch size
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = f'cuda:{gpu_id}'
    target_mpp = 0.5               # Target MPP (um/pixel)
    global_pool = True          # Whether to use global pooling in slide encoder
    tile_size = 224
    num_max_tiles = 50000  # Maximum number of tiles used per WSI
    
    # Load models
    print("Loading models...")
    tile_encoder, slide_encoder = load_models(global_pool=global_pool)
    
    # Process CSV file
    process_csv_file(top_dir, csv_path, output_dir, tile_encoder, slide_encoder,
                     target_mpp, tile_size, batch_size, num_max_tiles, device)
    
    print("Feature extraction completed!")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        csv_path = "datalist.csv"
        gpu_id = 0
    elif len(sys.argv) == 3:
        csv_path = sys.argv[1]
        gpu_id = int(sys.argv[2])
    else:
        print("Usage: python main.py <csv_path> <gpu_id>")
        sys.exit(1)

    main(csv_path, gpu_id)