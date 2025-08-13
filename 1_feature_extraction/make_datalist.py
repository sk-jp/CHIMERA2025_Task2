import glob

topdir = "/data/MICCAI2025_CHIMERA/task2/data"
subdirs = sorted(glob.glob(f"{topdir}/2*"))

#print("wsi_path,mask_path")
for subdir in subdirs:
    case_name = subdir.split("/")[-1]
    he_file = f"{subdir}/{case_name}_HE.tif"
    mask_file = f"{subdir}/{case_name}_HE_mask.tif"
    print(f"{he_file.replace(topdir+'/', '')},{mask_file.replace(topdir+'/', '')}")

