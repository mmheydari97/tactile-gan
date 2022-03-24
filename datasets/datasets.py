from datasets.PairedDataset import PairedDataset
from datasets.UnpairedDataset import UnpairedDataset

def get_dataset(img_dir, opt, size, flip, jitter):
    if opt.paired_dataset:
        dataset = PairedDataset(img_dir, size, flip, jitter)
    else:
        dataset = UnpairedDataset()
        
    return dataset