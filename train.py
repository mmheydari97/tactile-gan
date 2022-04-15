import torch
import torch.nn as nn

import argparse
import os
import json

from torch.utils.data import DataLoader

from statistics import mean
  
from torch.optim import lr_scheduler
import numpy as np
import time
from generators.generators import create_gen
from discriminators.discriminators import create_disc
from datasets.datasets import get_dataset
from util import set_requires_grad,init_weights, mkdir, VGGPerceptualLoss
from Tensorboard_Logger import Logger


class Train_Pix2Pix:
    '''
    GAN model for Pix2Pix implementation. Trains models, saves models, saves training results and details about the models.
    '''
    def __init__(self,opt,traindataset):
        
        #load in the datasets
        self.dataset = DataLoader(dataset=traindataset, batch_size=opt.batch_size, shuffle=True,num_workers=opt.threads)
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

        if opt.continue_training:
            checkpoint = torch.load(os.path.join(opt.dir,"models",opt.folder_load,opt.gen))

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

        for i in range(opt.total_iters):
            epoch = i + opt.epoch_count
            #monitor each minibatch loss
            lossdlist = []
            lossglist = []
            lossl1list = []
            lossperlist = []
            
            t1 = time.time()
            
            for j, batch in enumerate(self.dataset):
                if j % 100 == 0:
                    print("training epoch ",epoch,"batch", j,"/",len(self.dataset))
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
                    loss_D = (loss_D_fake + loss_D_real) * 0.5 * opt.lambda_GP
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
                
                loss_G_L1 = nn.L1Loss()(real_B,fake_B) #get per pixel L1 Loss
                lossl1list.append( loss_G_L1.item())

                if not(opt.loss == "wloss"):
                    loss_G = loss_G_GAN + loss_G_L1 * opt.lambda_A
                else:
                    loss_G = pred_fake.mean()*-1 #the Generator Loss in WGAn is different
                if opt.lambda_per != 0:
                    per_loss = perceptual.forward(fake_B, real_A, [0,1])
                    loss_G += per_loss * opt.lambda_per
                    lossperlist.append(per_loss.item())
                else:
                    lossperlist.append(0)
                
                ###########################
                # lossglist.append(loss_G.item())
                ###########################
                
                lossglist.append(loss_G_GAN.item())
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


            if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:                
                with torch.no_grad():
                    out1 = self.netG(self.atrain.to(self.device))
                    title= "Epoch "+str(epoch) +"Training"
                    self.writer.write_sketch_to_tb(out1.detach(),title) 
            
                    out2 = self.netG(self.atest.to(self.device))
                    title= "Epoch "+str(epoch)
                    self.writer.write_sketch_to_tb(out2.detach(),title)
                torch.save(self.netG.state_dict(), f"{opt.dir}/checkpoints/{opt.folder_save}/{opt.gen}_{epoch}")
                # torch.save(self.netD.state_dict(), f"{opt.dir}/checkpoints/{opt.folder_save}/{opt.disc}_{epoch}")            

            
        self.writer.plot_losses(self.gen_loss,self.disc_loss,self.l1_loss)
    
    @staticmethod
    def get_scheduler(optimizer):
        milestone = np.int16(np.linspace(opt.iter_constant, opt.total_iters, 11)[:-1])
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=list(milestone), gamma=0.8)
        return scheduler
        
    def save_model(self,modelpath):
        '''
        Saves the models as well as the optimizers
        '''
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
        

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="data directory")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
parser.add_argument("--test_batch_size", type=int, default=16, help="test batch size")
parser.add_argument("--input_dim", type=int, default=3, help="input depth size")
parser.add_argument("--output_dim", type=int, default=3, help="output depth size")
parser.add_argument("--gen_filters", type=int, default=64, help="starting filters for the generator")
parser.add_argument("--disc_filters", type=int, default=64, help="starting filters for the discriminator")
parser.add_argument("--epoch_count", type=int, default=1, help="starting epoch, useful if we're loading in a half trained model, we can change starting epoch")
parser.add_argument("--total_iters", type=int, help="total epochs we're training for")
parser.add_argument("--iter_constant", type=int, default=200, help="how many epochs we keep the learning rate constant")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--label_smoothing", default=False, action='store_true', help="if written, we will use one sided label smoothing")
parser.add_argument("--beta1", type=float, default=0.01, help="beta1 for our Adam optimizer")
parser.add_argument("--cuda", default=True, action='store_false', help="if written, we will not use gpu accelerated training")
parser.add_argument("--threads", type=int, default=8, help="cpu threads for loading the dataset")
parser.add_argument("--lambda_A", type=float, default=0.1, help="L1 lambda")
parser.add_argument("--lambda_per", type=float, default=0.1, help="perceptual lambda")
parser.add_argument("--lambda_GP", type=float, default=10, help="Gradient_penalty loss")
parser.add_argument("--norm", default="instance", help="normalization mode")
parser.add_argument("--gen", default="UNet++", choices=["Resnet", "UNet++", "UNet", "UNet-no-skips", "BCDUNet"], help="generator architecture")
parser.add_argument("--disc", default="Patch", choices=["Global", "Patch"], help="discriminator architecture")
parser.add_argument("--loss", default="ls", choices=["ls", "bce", "wloss"], help="loss function")
parser.add_argument("--no_aug", default=False, action='store_true', help="if written, we won't augment the dataset")
parser.add_argument("--folder_save", default="pix2pix", help="where we want to save the model to")
parser.add_argument("--folder_load", default="pix2pix", help="where we want to load the model from")
parser.add_argument("--checkpint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--continue_training", default=False, action='store_true', help="if written, we will load the weights for the network brfore training")

opt = parser.parse_args()


photo_path_train = os.path.join(opt.dir,"data","train", "source")
train_set = get_dataset(photo_path_train, opt, mode="train")

experiment = Train_Pix2Pix(opt,train_set)

checkpoint_path = os.path.join(opt.dir,"checkpoints",opt.folder_save)
mkdir(checkpoint_path)
save_path = os.path.join(opt.dir,"models",opt.folder_save)
mkdir(save_path)
model_path = os.path.join(save_path,"final_model.pth")
experiment.train(opt)

experiment.save_model(model_path)
experiment.save_arrays(save_path)
experiment.save_hyper_params(save_path,opt)
