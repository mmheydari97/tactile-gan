import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self,writer_title):
        title= 'runs/'+writer_title
        self.writer = SummaryWriter(title)
        
    def write_photo_to_tb(self,photo,img_label,nrows=8,step=None):
        grid = torchvision.utils.make_grid(photo,nrow=nrows)
        grid = grid.cpu().float().numpy()/2 + 0.5 #normalize from [-1,1] to [0,1]
        self.writer.add_image(img_label, grid,global_step = step)
    
    def write_sketch_to_tb(self,sketch,label,nrows=8,step=0):
        sketch = sketch[:,0,:,:].unsqueeze(1) #because
        grid = torchvision.utils.make_grid(sketch, nrow=nrows )
        grid = grid.cpu().float().numpy()/2 + 0.5
        self.writer.add_image(label, grid,global_step = step)
    
    def write_sketch_to_tb_2(self,sketch,label,nrows=8,step=0):
#         sketch = sketch[:,0,:,:].unsqueeze(1) #because
        grid = torchvision.utils.make_grid(sketch, nrow=nrows )
        grid = grid.cpu().float().numpy()/2 + 0.5
        self.writer.add_image(label, grid,global_step = step)
        
    #cant get this to work
    def log_loss(self,title,loss,epoch):
#         print("logging loss with epoch:", epoch)
#         self.writer.add_scalar(title,loss,global_step=epoch)
        self.writer.add_scalar(title,loss)
    
    def plot_losses(self, gen_loss, disc_loss,l1_loss):
        for i in range(len(gen_loss)):
            self.writer.add_scalar('Generator_loss', gen_loss[i], i)
            self.writer.add_scalar('Discriminator_loss', disc_loss[i], i)
            if len(l1_loss) != 0:
                self.writer.add_scalar('L!_loss', l1_loss[i], i)