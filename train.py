import argparse
import json
import os
from statistics import mean
import time

import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from tqdm import tqdm

from generators.generators import create_gen, GANLoss
from discriminators.discriminators import create_disc
from datasets.datasets import get_dataset
from util import set_requires_grad, init_weights, mkdir, perceptual_loss


class Train_Pix2Pix:
    '''
    GAN model for Pix2Pix implementation. Trains models, saves models, saves training results and details about the models.
    '''
    def __init__(self,opt,traindataset):
        
        #load in the datasets
        self.dataset = DataLoader(dataset=traindataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)

        #tensorflow logger
        self.device = torch.device("cuda:0" if opt.cuda else "cpu")

        #create generator and discriminator
        self.netG = create_gen(opt.gen, opt.input_dim, opt.output_dim)
        self.netG.to(self.device)
        init_weights(self.netG)

        self.netD = create_disc(opt.disc, opt.input_dim, opt.output_dim)
        self.netD.to(self.device)
        init_weights(self.netD)

     
        self.gan_loss = GANLoss(gan_mode=opt.loss, label_smoothing=opt.label_smoothing, tensor=torch.cuda.FloatTensor)
          
        #optimizers and attach them to schedulers that change learning rate
        self.schedulers = []
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.99))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.99))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        for optimizer in self.optimizers:
            self.schedulers.append(self.get_scheduler(optimizer))

        #logging each epoch losses
        self.gen_loss = []
        self.disc_loss = []
        self.l1_loss = []
        self.per_loss = []
        self.gp_loss = []


        if opt.continue_training:
            checkpoint = torch.load(os.path.join(opt.data.rsplit("/",1)[0],"models",opt.folder_load,"final_model.pth"))

            self.netG.load_state_dict(checkpoint["gen"])
            self.optimizer_G.load_state_dict(checkpoint["optimizerG_state_dict"])
            self.netD.load_state_dict(checkpoint["disc"])
            self.optimizer_D.load_state_dict(checkpoint["optimizerD_state_dict"])
        
    
    def train(self,opt):
        '''
        Starts the training process. The details and parts of the model were already initialized in __init__
        '''

        # load vgg if we want to use perceptual loss

        for i in range(opt.total_iters):
            epoch = i + opt.epoch_count
            #monitor each minibatch loss
            lossdlist = []
            lossglist = []
            lossl1list = []
            lossperlist = []
            lossgpdlist = []
            
            t1 = time.time()
            
            print("==training epoch ",epoch)
            for j, batch in enumerate(tqdm(self.dataset)):
                    
                real_A, real_B = batch[0].to(self.device), batch[1].to(self.device) #load in a batch of data

                # (generate fake images)
                fake_B = self.netG(real_A)
                
                #Optimize D ################################
                set_requires_grad(nets=self.netD, requires_grad=True) #optimize for D so create computation graph
                self.optimizer_D.zero_grad()
                
                pred_fake = self.netD(real_A, fake_B.detach()) #generate predictions on fake images
                pred_real = self.netD(real_A, real_B)
                
                
                loss_D_fake = self.gan_loss(pred_fake, False, for_discriminator=True).mean()

                #Get the second half of the loss. Compare real predictions to labels of 1        
                loss_D_real = self.gan_loss(pred_real, True, for_discriminator=True).mean()
                
                loss_D = (loss_D_fake + loss_D_real)/2
                
                lossdlist.append(loss_D.item())
                
                #now that we have the full loss we back propogate
                loss_D.backward()
                self.optimizer_D.step()

                d_regularize = (opt.d_reg_every!=0) and (epoch % opt.d_reg_every == 0) and (opt.lambda_gp!=0)
                if d_regularize:
                    self.optimizer_D.zero_grad()
                    gp_loss = self.gradient_penalty(real_A, real_B, fake_B, lambda_gp=opt.lambda_gp)
                    gp_loss.backward(retain_graph=True)
                    self.optimizer_D.step()
                    lossgpdlist.append(gp_loss.item())
                else:
                    lossgpdlist.append(0)


                # Optimize G #####################################
                set_requires_grad(nets=self.netD, requires_grad=False) #dont want the discriminator to update weights this round
                self.optimizer_G.zero_grad()
                pred_fake = self.netD(real_A, fake_B) #generate D predictions of fake images

                loss_G_GAN = self.gan_loss(pred_fake, True, for_discriminator=False).mean() #We feed it real_labels as G is trying fool the discriminator
                lossglist.append(loss_G_GAN.item())
                
                loss_G_L1 = nn.L1Loss()(real_B,fake_B) #get per pixel L1 Loss
                lossl1list.append(loss_G_L1.item())

                loss_G = loss_G_GAN + loss_G_L1 * opt.lambda_a
                if opt.lambda_per != 0:
                    features_fake = self.netD.get_intermediate_output()
                    _ = self.netD(real_A, real_B)
                    features_real = self.netD.get_intermediate_output()
                    per_loss = perceptual_loss(features_real, features_fake, weights=opt.w_per)
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
            t2 = time.time()
            diff = t2-t1
            print(f"\tloss functions => D:{mean(lossdlist):.5f}, G:{mean(lossglist):.5f}, L1:{mean(lossl1list):.5f}, gp:{mean(lossgpdlist):.5f}, per:{mean(lossperlist):.5f}")
            print(f'\tlearing rate: {lr:.5f}')
            print(f"\ttook {diff:.2f} seconds")
            print(f"\tapproximately {diff*(opt.total_iters - epoch):.2f} seconds left")

            self.gen_loss.append(mean(lossglist))
            self.disc_loss.append(mean(lossdlist))
            self.l1_loss.append(mean(lossl1list))
            self.per_loss.append(mean(lossperlist))
            self.gp_loss.append(mean(lossgpdlist))
            if opt.checkpoint_interval != -1 and epoch%opt.checkpoint_interval == 0:
                self.save_model(f"{opt.data.rsplit('/',1)[0]}/checkpoints/{opt.folder_save}/model_{epoch}.pth")


    @staticmethod
    def get_scheduler(optimizer):
        milestone = np.int16(np.linspace(opt.iter_constant, opt.total_iters, 11)[:-1])
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=list(milestone), gamma=0.8)
        return scheduler
    
    def gradient_penalty(self, real_img, real_mask, fake_mask, type='mixed', constant=1.0, lambda_gp=1.0):
        if lambda_gp > 0.0:
            if type == 'real':
                interpolates = real_mask
            elif type == 'fake':
                interpolates = fake_mask
            elif type == 'mixed':
                alpha = (torch.rand(real_mask.size(0), 1, device=self.device)+1)/2
                alpha = alpha.expand(real_mask.shape[0], real_mask.nelement() // real_mask.shape[0]).contiguous().view(*real_mask.shape)
                interpolates = alpha * real_mask + ((1 - alpha) * fake_mask)
            else:
                raise NotImplementedError(f'{type} not implemented')
            interpolates.requires_grad_(True)
            pred = self.netD(real_img, interpolates)
            gradients = torch.autograd.grad(outputs=pred, inputs=interpolates,
                                            grad_outputs=torch.ones(pred.size()).to(self.device),
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True)
            gradients = gradients[0].view(real_mask.size(0),-1)                
            res = (((gradients+1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
            
            return res
        else:
            return 0.0


    def save_model(self,modelpath):
        '''
        Saves the models as well as the optimizers
        '''
        if not os.path.exists(modelpath.rsplit('/',1)[0]):
            mkdir(modelpath.rsplit('/',1)[0])
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
        np.save( os.path.join(path,"perloss"),np.asarray(self.per_loss))
        np.save( os.path.join(path,"gploss"),np.asarray(self.gp_loss))

        
    def save_hyper_params(self,folderpath,opt):
        '''
        We need to load back in the details of the model when we test so we save these as well
        '''
        with open(os.path.join(folderpath,'params.txt'), 'w') as file:
             file.write(json.dumps(opt.__dict__)) 
        

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./data", help="dataset directory")
parser.add_argument("--batch_size", type=int, default=4, help="training batch size")
parser.add_argument("--input_dim", type=int, default=3, help="input depth size")
parser.add_argument("--output_dim", type=int, default=3, help="output depth size")
parser.add_argument("--epoch_count", type=int, default=1, help="starting epoch, useful if we're loading in a half trained model, we can change starting epoch")
parser.add_argument("--total_iters", type=int, default=135, help="total epochs we're training for")
parser.add_argument("--iter_constant", type=int, default=25, help="how many epochs we keep the learning rate constant")
parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
parser.add_argument("--label_smoothing", default=True, action='store_true', help="if written, we will not use one sided label smoothing")
parser.add_argument("--beta1", type=float, default=0.01, help="beta1 for our Adam optimizer")
parser.add_argument("--cuda", default=True, action='store_false', help="if written, we will not use gpu accelerated training")
parser.add_argument("--threads", type=int, default=8, help="cpu threads for loading the dataset")
parser.add_argument("--lambda_a", type=float, default=5, help="L1 loss coefficient")
parser.add_argument('--lambda_gp', type=float, default=0.1, help="gradient penalty coefficient")
parser.add_argument("--lambda_per", type=float, default=0.2, help="perceptual loss coefficient")
parser.add_argument('--w_per', nargs=4, type=float, default=[0,.1,.3,.6], help='perceptual weights')
parser.add_argument("--gen", default="UNet++", choices=["UNet++", "UNet"], help="generator architecture")
parser.add_argument("--disc", default="Patch", choices=["Global", "Patch"], help="discriminator architecture")
parser.add_argument("--loss", default="ls", choices=["ls", "ce", "w", "hinge"], help="loss function for ganloss")
parser.add_argument("--no_aug", default=False, action='store_true', help="if written, we won't augment the dataset")
parser.add_argument("--folder_save", default="pix2obj", help="where we want to save the model to")
parser.add_argument("--folder_load", default="pix2obj", help="where we want to load the model from")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--continue_training", default=False, action='store_true', help="if written, we will load the weights for the network brfore training")
parser.add_argument('--d_reg_every', type=int, default=4, help='set how frequently we regularize the discriminator')

opt = parser.parse_args()

 
photo_path_train = os.path.join(opt.data, "train", "source")
train_set = get_dataset(photo_path_train, opt, mode='train')

experiment = Train_Pix2Pix(opt,train_set)

checkpoint_path = os.path.join(os.getcwd(),"checkpoints",opt.folder_save)
mkdir(checkpoint_path)
save_path = os.path.join(os.getcwd(),"models",opt.folder_save)
print(save_path)
mkdir(save_path)
model_path = os.path.join(save_path,"final_model.pth")
experiment.train(opt)

experiment.save_model(model_path)
experiment.save_arrays(save_path)
experiment.save_hyper_params(save_path,opt)
