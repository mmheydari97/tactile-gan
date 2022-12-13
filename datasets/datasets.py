from datasets.PairedDataset import PairedDataset

def get_dataset(img_dir, opt, mode='train'):
    dataset = PairedDataset(img_dir, mode=mode, aug=not opt.no_aug, target=opt.target)
        
    return dataset
