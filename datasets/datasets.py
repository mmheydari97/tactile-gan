from datasets.PairedDataset import PairedDataset
from datasets.UnpairedDataset import UnpairedDataset

def get_dataset(photo_path,sketch_path,opt,flip,jitter,erase, colored_s):
    if opt.paired_dataset:
        dataset = PairedDataset(photo_path,sketch_path,opt.dataset_name,size=256,flip=flip,jitter=jitter,erase=erase, colored_sketch=colored_s)
    else:
        dataset = UnpairedDataset(photo_path,sketch_path,opt.dataset_name,size=256,erase=erase)
        
    return dataset