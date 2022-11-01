from datasets.PairedDataset import PairedDataset

def get_dataset(img_dir, opt, mode='train', gt=True):
    dataset = PairedDataset(img_dir, mode=mode, aug=not opt.no_aug, gt=gt)
        
    return dataset
