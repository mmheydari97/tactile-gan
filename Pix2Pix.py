#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# import os

# if os.getcwd() != '/content/drive/My Drive/Pix2Pix':
#     drive.mount('/content/drive')
#     get_ipython().run_line_magic('cd', 'drive/MyDrive/Pix2Pix/')
# get_ipython().run_line_magic('ls', '')


# In[4]:


import torch
import torch.nn as nn

import argparse
import os
import sys
from math import log10
import json

import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


from statistics import mean


from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import time
from generators.generators import create_gen
from discriminators.discriminators import create_disc
from datasets.datasets import get_dataset
from util import ImagePool, set_requires_grad,tensor_to_plt,init_weights, mkdir
from Tensorboard_Logger import Logger


# In[5]:


def get_scheduler(optimizer ):
    '''
    Learning rate scheduler. We want to start off at a constant rate and slowly decay
    '''
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.iter_constant) / float(opt.iter_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


class Train_Pix2Pix:
    '''
    GAN model for Pix2Pix implementation. Trains models, saves models, saves training results and details about the models.
    '''
    def __init__(self,opt,traindataset,testdataset):
        
        #load in the datasets
        self.dataset = DataLoader(dataset=traindataset, batch_size=opt.batch_size, shuffle=True,num_workers=opt.threads)
        self.test_set = DataLoader(dataset=testdataset, batch_size=opt.test_batch_size, shuffle=False,num_workers=opt.threads)
        self.atest, self.btest = next(iter(self.test_set))
        self.dataviz = DataLoader(dataset=traindataset, batch_size=opt.batch_size, shuffle=False,num_workers=opt.threads)
        self.atrain, self.btrain = next(iter(self.dataviz))

        #tensorflow logger
        self.device = torch.device("cuda:0" if opt.cuda else "cpu")
        self.writer = Logger(opt.folder_name)
        self.writer.write_photo_to_tb(self.atest,"photos test")
        self.writer.write_sketch_to_tb(self.btest,"sketches test")
        self.writer.write_photo_to_tb(self.atrain,"photos train")
        self.writer.write_sketch_to_tb(self.btrain,"sketches train")


        #create generator and discriminator
        self.netG = create_gen(opt.gen,opt.input_dim,opt.output_dim,opt.gen_filters,opt.norm)
        self.netG.to(self.device)
        init_weights(self.netG)
        use_sigmoid = True if opt.loss == "bce" else False
        self.netD = create_disc(opt.disc,opt.input_dim+opt.output_dim,use_sigmoid)
        self.netD.to(self.device)
        init_weights(self.netD)
        #set the GAN adversarial loss
        if opt.loss =="bce":
            self.criterion = nn.BCELoss().to(self.device) 
        elif opt.loss == "ls":
            self.criterion = nn.MSELoss().to(self.device)
        elif opt.loss == "wloss":
            self.criterion = self.get_w_loss
            
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
            self.schedulers.append(get_scheduler(optimizer))

        #logging each epoch losses
        self.gen_loss = []
        self.disc_loss = []
        self.l1_loss = []

    def train(self,opt):
        '''
        Starts the training process. The details and parts of the model were already initialized in __init__
        '''
        for epoch in range(opt.epoch_count, opt.total_iters + 1):

            #monitor each minibatch loss
            lossdlist = []
            lossglist = []
            lossl1list = []
            
            t1 = time.time()
            
            for i, batch in enumerate(self.dataset):
                if i % 10 == 0:
                    print("training epoch ",epoch,"batch", i,"/",len(self.dataset))
                real_A, real_B = batch[0].to(self.device), batch[1].to(self.device) #load in a batch of data

                # (generate fake images)
                fake_B = self.netG(real_A)

                #Optimize D ################################
                set_requires_grad(nets=self.netD, requires_grad=True) #optimize for D so create computation graph
                self.optimizer_D.zero_grad()
                
                fake_AB =torch.cat((real_A, fake_B), 1).detach() #we detach because when we train discriminator we don't want to back propogate the generator
                pred_fake = self.netD(fake_AB.detach()) #generate predictions on fake images
                
                #create labels, either 1's or 0.9 if we're smooting.  
                if opt.label_smoothing:
                    real_labels = torch.normal(.9, .02, size=pred_fake.size()).to(self.device)
                else:
                    real_labels = torch.full(pred_fake.size(),self.real_label_value).to(self.device)
                fake_labels = torch.full(pred_fake.size(),self.fake_label_value).to(self.device)
                
                
                if not(opt.loss == "wloss"):
                    #Get the first half of the loss. Comparing fake predictions to labels of 0
                    loss_D_fake = self.criterion(pred_fake, fake_labels)

                #Get the second half of the loss. Compare real predictions to labels of 1
                #Here we have 5 real sketches per image, so we loop it 5 times.
                n = real_B.shape[1]//opt.output_dim
                loss_D_real_set = torch.empty(n, device=self.device)
                for i in range(n):
                    sel_B = real_B[:, i:i+opt.output_dim, :, :] #.unsqueeze(1)
        
                    real_AB = torch.cat((real_A, sel_B), 1)
                    pred_real = self.netD(real_AB)
                    if not(opt.loss == "wloss"): 
                        loss_D_real_set[i] = self.criterion(pred_real, real_labels)
                    else: #if we're using Wassersetin loss
                        loss_D_real_set[i] = self.criterion(real_AB,fake_AB, pred_fake, pred_real, opt.lambda_GP)        
                loss_D_real = torch.mean(loss_D_real_set)
                
                if not(opt.loss == "wloss"): 
                    loss_D = (loss_D_fake + loss_D_real) * 0.5 * opt.lambda_G
                else:
                    loss_D = loss_D_real
                
                lossdlist.append(loss_D.item())
                
                #now that we have the full loss we back propogate
                loss_D.backward(retain_graph=True if opt.loss == "wloss" else False)
                self.optimizer_D.step()

                # Optimize G #####################################
                set_requires_grad(nets=self.netD, requires_grad=False) #dont want the discriminator to update weights this round
                self.optimizer_G.zero_grad()

                fake_AB = torch.cat((real_A, fake_B), 1)
                pred_fake = self.netD(fake_AB) #generate D predictions of fake images
                if not(opt.loss == "wloss"):
                    loss_G_GAN = self.criterion(pred_fake, real_labels) #We feed it real_labels as G is trying fool the discriminator
                
                loss_G_L1 = self.get_l1_loss(real_B,fake_B) #get per pixel L1 Loss
                lossl1list.append( loss_G_L1.item())

                if not(opt.loss == "wloss"):
                    loss_G = loss_G_GAN + loss_G_L1 * opt.lambda_A
                else:
                    loss_G = pred_fake.mean()*-1 #the Generator Loss in WGAn is different
                    
                lossglist.append(loss_G.item())
                loss_G.backward()
                self.optimizer_G.step()

            #update_learning_rate()
            for scheduler in self.schedulers:
                scheduler.step()
            lr = self.optimizers[0].param_groups[0]['lr']
            print('learning rate = %.7f' % lr)
            t2 = time.time()
            diff = t2-t1
            print("iteration:",epoch,"loss D:", mean(lossdlist),"loss G:", mean(lossglist))
            print("Took ", diff, "seconds")
            print("Estimated time left:", diff*(opt.total_iters - epoch))

            self.gen_loss.append(mean(lossglist))
            self.disc_loss.append(mean(lossdlist))
            self.l1_loss.append(mean(lossl1list))


            if epoch % 1 == 0:                
                with torch.no_grad():
                    out1 = self.netG(self.atrain.to(self.device))
                    title= "Epoch "+str(epoch) +"Training"
                    self.writer.write_sketch_to_tb(out1.detach(),title) 
            
                    out2 = self.netG(self.atest.to(self.device))
                    title= "Epoch "+str(epoch)
                    self.writer.write_sketch_to_tb(out2.detach(),title)
            

            
        self.writer.plot_losses(self.gen_loss,self.disc_loss,self.l1_loss)
    
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
        
    def get_w_loss(self,real_AB,fake_AB, pred_fake, pred_real, gp_lambda):
        '''
        Wasserstein Loss with Gradient Penalty
        '''
        epsilon = torch.rand(len(real_AB), 1, 1, 1, requires_grad=True).to(self.device)
        interpolated = real_AB*epsilon + fake_AB * (1 - epsilon)
        mixed_pred = self.netD(interpolated)
        gradient = torch.autograd.grad(inputs=interpolated,outputs=mixed_pred,
                    grad_outputs=torch.ones_like(mixed_pred), create_graph=True,retain_graph=True,)[0]
        
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        gp = (gradient_norm -1)**2
        gp = gp.mean()
        
        loss = pred_fake.mean() - pred_real.mean() + gp* gp_lambda
        return loss
    
    def get_l1_loss(self,real_B,fake_B):
        '''
        L1 Loss for Pix2Pix. Compares the Pixel Loss between the fake image and it's closest real image
        '''
        if real_B.shape[1] != fake_B.shape[1]:
            fake_B = fake_B.expand(-1, real_B.shape[1], -1, -1)
        L1 = torch.abs(fake_B - real_B)
        L1 = L1.view(-1, real_B.shape[1], real_B.shape[2]*real_B.shape[3])
        L1 = torch.mean(L1, 2)
        min_L1, min_idx = torch.min(L1, 1)
        loss_G_L1 = torch.mean(min_L1)
        return loss_G_L1
        


# In[6]:


class Args():
    '''
    We set model details as a class that we can pass around
    '''
    def __init__(self):
        self.batch_size = 16 
        self.test_batch_size = 16
        self.input_dim = 3
        self.output_dim = 1
        self.gen_filters =64 #starting filters for the generator
        self.disc_filters =64 #starting filters for the discriminator
        self.epoch_count =1 #starting epoch, useful if we're loading in a half trained model, we can change starting epoch
        self.total_iters=3 #total epochs we're training for
        self.iter_constant = 200 #how many epochs we keep the learning rate constant
        self.iter_decay = 200 #when we start decaying the learning rate
        self.lr = 0.0002
        self.label_smoothing = True #if True, we use one sided label smoothing
        self.beta1 = 0.5 # beta1 for our Adam optimizer
        self.cuda = True
        self.threads = 8
        self.lambda_A = 0 #L1 lambda
        self.lambda_GP = 10 #Gradient_penalty loss
        self.use_ls = True
        self.resblocks = 9 #number of resblocks in bottleneck if we're using resnet generator
        self.norm = "instance"
        self.gen = "UNet" # Resnet, UNet++, UNet, UNet-no-skips
        self.disc= "Global" #Global, Patch
        self.loss = "wloss" #ls, bce, wloss
        self.paired_dataset = True
        self.dataset_name = "tactile" # "tactile", "aligned" or "sketchy"
        self.flip = True #image augementation flip horizontally
        self.jitter = True #image augementation vary color, brightness and contrast
        self.erase = True #image augementation randomly erase a portion of input image
        self.folder_name = "wgan_sketchy_resnet" #where we want to save the model to
    
    
opt = Args()

photo_path_train = os.path.join(str(sys.argv[1]),"data",opt.dataset_name,"train", "photo")
sketch_path_train = os.path.join(str(sys.argv[1]),"data",opt.dataset_name,"train", "sketch")
train_set = get_dataset(photo_path_train,sketch_path_train, opt,flip=True,jitter=True,erase= False)

photo_path_test = os.path.join(str(sys.argv[1]),"data",opt.dataset_name,"test", "photo")
sketch_path_test = os.path.join(str(sys.argv[1]),"data",opt.dataset_name,"test", "sketch")
testing_set =  get_dataset(photo_path_test,sketch_path_test, opt,flip=False,jitter=False,erase= False)


# In[7]:


exps = [opt]
for option in exps:
    experiment = Train_Pix2Pix(option,train_set,testing_set)
    experiment.train(option)
    folderpath = os.path.join(str(sys.argv[1]),"models",option.folder_name)
    model_path = os.path.join(folderpath,option.gen)
    experiment.save_model(folderpath,model_path)
    experiment.save_arrays(folderpath)
    experiment.save_hyper_params(folderpath,opt)
