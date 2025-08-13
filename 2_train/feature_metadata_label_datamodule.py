import lightning.pytorch as pl
from torch.utils.data import DataLoader
from feature_metadata_label_dataset import FeatureMetadataLabelDataset


class FeatureMetadataLabelDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(FeatureMetadataLabelDataModule, self).__init__()

        # configs
        self.cfg = cfg

        self.dataset = {}
        self.dataset['train'] = FeatureMetadataLabelDataset(
            cfg.Data.dataset.top_dir,
            cfg.Data.dataset.train_datalist,
            cfg.Data.dataset.num_classes)
        self.dataset['valid'] = FeatureMetadataLabelDataset(
            cfg.Data.dataset.top_dir,
            cfg.Data.dataset.valid_datalist,
            cfg.Data.dataset.num_classes)

    # call once from main process
    def prepare_data(self):
        pass
 
    # call from Trainer.fit() and Trainer.test()
    def setup(self, stage=None):
        pass
    
    # call in Trainer.fit()
    def train_dataloader(self):
        train_loader = DataLoader(
            self.dataset['train'],
            batch_size=self.cfg.Data.dataloader.train.batch_size,
            shuffle=self.cfg.Data.dataloader.train.shuffle,
            num_workers=self.cfg.Data.dataloader.train.num_workers,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False
        )
        return train_loader

    # call in Trainer.fit() and Trainer.validate()
    def val_dataloader(self):
        val_loader = DataLoader(
            self.dataset['valid'],
            batch_size=self.cfg.Data.dataloader.valid.batch_size,
            shuffle=self.cfg.Data.dataloader.valid.shuffle,
            num_workers=self.cfg.Data.dataloader.valid.num_workers,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False
        )
        return val_loader

