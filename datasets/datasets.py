from datasets.PairedDataset import PairedDataset

def get_dataset(img_dir, opt, **kwargs):
    dataset = PairedDataset(img_dir, **kwargs)
        
    return dataset
