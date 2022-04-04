import torch
import torch.nn as nn

import argparse
import os
import json

import torch.optim as optim
from torch.utils.data import DataLoader

from statistics import mean

from torch.nn import init
from torch.optim import lr_scheduler
import numpy as np
import time
from generators.generators import create_gen
from discriminators.discriminators import create_disc
from datasets.datasets import get_dataset
from util import set_requires_grad, init_weights, mkdir, VGGPerceptualLoss


class Train_Pix2Pix:
    '''
    GAN model for Pix2Pix implementation. Trains models, saves models, saves training results and details about the models.
    '''
    def __init__(self,opt,traindataset):
        
        #load in the datasets
        self.dataset = DataLoader(dataset=traindataset, batch_size=opt.batch_size, shuffle=True,num_workers=opt.threads)

        #tensorflow logger
        self.device = torch.device("cuda:0" if opt.cuda else "cpu")

        #create generator and discriminator
        self.netG = create_gen(opt.gen, opt.input_dim, opt.output_dim)
        self.netG.to(self.device)
        init_weights(self.netG)

        self.netD = create_disc(opt.disc, opt.input_dim, opt.output_dim)
        self.netD.to(self.device)
        init_weights(self.netD)

     
        self.criterion = nn.MSELoss().to(self.device)
          
        self.real_label_value = 1.0
        self.fake_label_value = 0.0
        
        #optimizers and attach them to schedulers that change learning rate
        self.schedulers = []
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        for optimizer in self.optimizers:
            self.schedulers.append(self.get_scheduler(optimizer))

        #logging each epoch losses
        self.gen_loss = []
        self.disc_loss = []
        self.l1_loss = []
        self.per_loss = []

        if opt.continue_training:
            checkpoint = torch.load(os.path.join(opt.dir,"models",opt.folder_name,opt.gen))

            self.netG.load_state_dict(checkpoint["gen"])
            self.optimizer_G.load_state_dict(checkpoint["optimizerG_state_dict"])
            self.netD.load_state_dict(checkpoint["disc"])
            self.optimizer_D.load_state_dict(checkpoint["optimizerD_state_dict"])
        
    
    def train(self,opt):
        '''
        Starts the training process. The details and parts of the model were already initialized in __init__
        '''

        # load vgg if we want to use perceptual loss
        if opt.lambda_per != 0:
            perceptual = VGGPerceptualLoss(resize=True)

        for epoch in range(opt.epoch_count, opt.total_iters):

            #monitor each minibatch loss
            lossdlist = []
            lossglist = []
            lossl1list = []
            lossperlist = []
            
            t1 = time.time()
            
            for i, batch in enumerate(self.dataset):
                if i % 100 == 0:
                    print("training epoch ",epoch,"batch", i,"/",len(self.dataset))
                real_A, real_B = batch[0].to(self.device), batch[1].to(self.device) #load in a batch of data

                # (generate fake images)
                fake_B = self.netG(real_A)
                
                #Optimize D ################################
                set_requires_grad(nets=self.netD, requires_grad=True) #optimize for D so create computation graph
                self.optimizer_D.zero_grad()
                
                pred_fake = self.netD(real_A, fake_B.detach()) #generate predictions on fake images
                
                #create labels, either 1's or 0.9 if we're smooting.  
                if opt.label_smoothing:
                    real_labels = torch.normal(.9, .02, size=pred_fake.size()).to(self.device)
                else:
                    real_labels = torch.full(pred_fake.size(),self.real_label_value).to(self.device)
                fake_labels = torch.full(pred_fake.size(),self.fake_label_value).to(self.device)

                
                loss_D_fake = self.criterion(pred_fake, fake_labels)

                #Get the second half of the loss. Compare real predictions to labels of 1
        
                pred_real = self.netD(real_A, real_B)
                loss_D_real = self.criterion(pred_real, real_labels)
                
                loss_D = (loss_D_fake + loss_D_real)/2
                
                lossdlist.append(loss_D.item())
                
                #now that we have the full loss we back propogate
                loss_D.backward(retain_graph=False)
                self.optimizer_D.step()

                # Optimize G #####################################
                set_requires_grad(nets=self.netD, requires_grad=False) #dont want the discriminator to update weights this round
                self.optimizer_G.zero_grad()

                pred_fake = self.netD(real_A, fake_B) #generate D predictions of fake images
                loss_G_GAN = self.criterion(pred_fake, real_labels) #We feed it real_labels as G is trying fool the discriminator
                lossglist.append(loss_G_GAN.item())
                
                loss_G_L1 = nn.L1Loss()(real_B,fake_B) #get per pixel L1 Loss
                lossl1list.append(loss_G_L1.item())

                loss_G = loss_G_GAN + loss_G_L1 * opt.lambda_A
                if opt.lambda_per != 0:
                    per_loss = perceptual.forward(fake_B, real_A, [0,1])
                    loss_G += per_loss * opt.lambda_per
                    lossperlist.append(per_loss.item())
                else:
                    lossperlist.append(0)
                
                
                loss_G.backward()
                self.optimizer_G.step()

            #update_learning_rate()
            for scheduler in self.schedulers:
                scheduler.step()
            lr = self.optimizers[0].param_groups[0]['lr']
            print('learning rate = %.7f' % lr)
            t2 = time.time()
            diff = t2-t1
            print("iteration:",epoch,"loss D:", mean(lossdlist),"loss G:", mean(lossglist), "loss L1:", mean(lossl1list), "loss per:", mean(lossperlist))
            print("Took ", diff, "seconds")
            print("Estimated time left:", diff*(opt.total_iters - epoch))

            self.gen_loss.append(mean(lossglist))
            self.disc_loss.append(mean(lossdlist))
            self.l1_loss.append(mean(lossl1list))
            self.per_loss.append(mean(lossperlist))
            if opt.checkpoint_interval != -1 and epoch+1%opt.checkpoint_interval == 0:
                torch.save(self.netG.state_dict(), f"{opt.dir}/checkpoints/{opt.folder_save}/gen_{epoch}")
                torch.save(self.netD.state_dict(), f"{opt.dir}/checkpoints/{opt.folder_save}/disc_{epoch}")

    @staticmethod
    def get_scheduler(optimizer):

        milestone = np.int16(np.linspace(opt.iter_constant, opt.total_iters, 11)[:-1])
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=list(milestone), gamma=0.8)
        return scheduler
                
    def save_model(self,folderpath,modelpath):
        '''
        Saves the models as well as the optimizers
        '''
        mkdir(folderpath)
        torch.save({
            'gen': self.netG.state_dict(),
            'disc': self.netD.state_dict(),
            'optimizerG_state_dict': self.optimizer_G.state_dict(),
            'optimizerD_state_dict': self.optimizer_D.state_dict(),
            
            }, modelpath)
        
    def save_arrays(self,path):
        '''
        save the losses to numpy arrays
        '''
        np.save( os.path.join(path,"genloss"),np.asarray(self.gen_loss))
        np.save( os.path.join(path,"discloss"),np.asarray(self.disc_loss))
        np.save( os.path.join(path,"l1loss"),np.asarray(self.l1_loss))
        
    def save_hyper_params(self,folderpath,opt):
        '''
        We need to load back in the details of the model when we test so we save these as well
        '''
        with open(os.path.join(folderpath,'params.txt'), 'w') as file:
             file.write(json.dumps(opt.__dict__)) 
        

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="data directory")
parser.add_argument("--batch_size", type=int, default=1, help="training batch size")
parser.add_argument("--input_dim", type=int, default=3, help="input depth size")
parser.add_argument("--output_dim", type=int, default=4, help="output depth size")
parser.add_argument("--epoch_count", type=int, default=0, help="starting epoch, useful if we're loading in a half trained model, we can change starting epoch")
parser.add_argument("--total_iters", type=int, help="total epochs we're training for")
parser.add_argument("--iter_constant", type=int, default=200, help="how many epochs we keep the learning rate constant")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--label_smoothing", default=False, action='store_true', help="if written, we will not use one sided label smoothing")
parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for our Adam optimizer")
parser.add_argument("--cuda", default=True, action='store_false', help="if written, we will not use gpu accelerated training")
parser.add_argument("--threads", type=int, default=8, help="cpu threads for loading the dataset")
parser.add_argument("--lambda_A", type=float, default=5, help="L1 lambda")
parser.add_argument("--lambda_per", type=float, default=0.0, help="perceptual lambda")
parser.add_argument("--gen", default="UNet++", choices=["UNet++", "UNet"], help="generator architecture")
parser.add_argument("--disc", default="Patch", choices=["Global", "Patch"], help="discriminator architecture")
parser.add_argument("--no_aug", default=False, action='store_true', help="if written, we won't augment the dataset")
parser.add_argument("--folder_save", default="pix2seg", help="where we want to save the model to")
parser.add_argument("--folder_load", default="pix2seg", help="where we want to load the model from")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--continue_training", default=False, action='store_true', help="if written, we will load the weights for the network brfore training")

opt = parser.parse_args()

 
photo_path_train = os.path.join(opt.dir, "data", "train", "source")
train_set = get_dataset(photo_path_train, opt, mode='train')

experiment = Train_Pix2Pix(opt,train_set)

checkpoint_path = os.path.join(f"{opt.dir}","checkpoints", f"{opt.folder_save}/")
mkdir(checkpoint_path)
folderpath = os.path.join(opt.dir,"models",opt.folder_save)
model_path = os.path.join(folderpath,opt.gen)
experiment.train(opt)

experiment.save_model(folderpath,model_path)
experiment.save_arrays(folderpath)
experiment.save_hyper_params(folderpath,opt)


