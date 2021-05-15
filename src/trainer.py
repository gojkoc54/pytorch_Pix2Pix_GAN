import torch 
from torch import nn 
from abc import ABC
import sys

from models import Pix2Pix

class Trainer(ABC):
    
    def __init__(self, params):
        
        try:
            # Pix2Pix class instance
            self.model = params['model']
            self.G_optimizer = params['G_optimizer']
            self.D_optimizer = params['D_optimizer']
            self.GAN_criterion = params['GAN_criterion']
            
            self.starting_epoch = params['starting_epoch']
            self.num_epochs = params['num_epochs']
            self.batch_size = params['batch_size']
            self.device = params['device']

        except KeyError as err:
            print("You are missing some input parameters! \n\n")
            print("Error message: ", err.msg)
            sys.exit(1)




class Pix2PixTrainer(Trainer):

    def __init__(self, params):
        super(Pix2PixTrainer, self).__init__(params)
        
        self.L1_criterion = params['L1_criterion']
        self.lambda_L1 = params['lambda_L1']
        

    def calculate_GAN_loss(self, preds, target_is_real=True):
        
        if target_is_real:
            labels = torch.ones_like(preds)
        else:
            labels = torch.zeros_like(preds)

        GAN_loss = self.GAN_criterion(preds, labels)

        return GAN_loss         


    def backward_D(self, real_in, real_out, fake_out):

        # -- Calculate GAN loss for the fake concatenation --
        # Feed the [G(x) + x] concatenation to D, telling D that it's fake
        fake_concat = torch.cat([fake_out, real_in], dim=1)

        # Since fake_out was generated by G, we have to detach it 
        # before feeding it to D, in order to stop backprop through G
        preds_fake = self.model.D(fake_concat.detach())

        GAN_loss_fake = self.calculate_GAN_loss(preds_fake, target_is_real=False)

        # -- Calculate GAN loss for the real concatenation --
        # Feed the [y + x] concatenation to D, telling D that it's real
        real_concat = torch.cat([real_out, real_in], dim=1)
        preds_real = self.model.D(real_concat)

        GAN_loss_real = self.calculate_GAN_loss(preds_real, target_is_real=True)

        # -- Take the mean loss and backprop it --
        GAN_loss = (GAN_loss_fake + GAN_loss_real) / 2
        GAN_loss.backward()


    def backward_G(self, real_in, real_out, fake_out):

        # -- Calculate the GAN loss --
        # Feed the [G(x) + x] concatenation to D, telling D that it's TRUE, 
        # because G wants to 'fool' D, so it's usefull to see how far he is from that.
        # The farther G(x) is from y, the greater D's loss is going to be.
        fake_concat = torch.cat([fake_out, real_in], dim=1)
        preds_fake = self.model.D(fake_concat)

        GAN_loss = self.calculate_GAN_loss(preds_fake, target_is_real=False)

        # -- Calculate the Pixel Distance Loss --
        # Compare the real output [y] and G's output [G(x)]
        L1_loss = self.L1_criterion(fake_out, real_out) * self.lambda_L1

        # -- Combine the losses into one and backprop it --
        G_loss = GAN_loss + L1_loss
        G_loss.backward()



    def train(self, train_loader, val_loader, writer):

        for self.epoch in range(self.starting_epoch, self.num_epochs):
            
            self.model.set_mode(G_mode='train', D_mode='train')

            i = 0

            for data in train_loader:
                
                real_in, real_out = \
                    data['in'].to(self.device), data['out'].to(self.device)

                # -- Generator forward --
                fake_out = self.model.G(real_in)

                # -- Update Discriminator --
                # Enable backprop through D (G is enabled all the time)
                self.model.set_requires_grad(self.model.D, True)
                self.D_optimizer.zero_grad()
                # Calculate D loss, then compute D's gradients
                # Do not compute G's gradients ! ==>> detach fake_out
                self.backward_D(real_in, real_out, fake_out)
                self.D_optimizer.step()

                # -- Update Generator --
                # Disable backprop through D
                self.model.set_requires_grad(self.model.D, False)
                self.G_optimizer.zero_grad()
                # Calculate G loss, then compute G's gradients 
                self.backward_G(real_in, real_out, fake_out)
                self.G_optimizer.step()

                


