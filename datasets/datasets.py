from datasets.PairedDataset import PairedDataset

def get_dataset(img_dir, opt):
    dataset = PairedDataset(img_dir, aug=opt.aug)
        
    return dataset
