import torch 
from torch import nn 
from abc import ABC
import sys

from models import Pix2Pix

class Trainer(ABC):
    
    def __init__(self, **kwargs):
        
        try:
            # Pix2Pix class instance
            self.model = kwargs['model']
            self.G_optimizer = kwargs['G_optimizer']
            self.D_optimizer = kwargs['D_optimizer']
            self.GAN_loss = kwargs['GAN_loss']
            self.pixel_loss = kwargs['pixel_loss']

            self.num_epochs = kwargs['epochs']
            self.batch_size = kwargs['batch_size']
            self.device = kwargs['device']

        except KeyError as err:
            print("You are missing some input parameters!")
            print("Error message: ", err.msg)
            sys.exit(1)




class Pix2PixTrainer(Trainer):

    def __init__(self):
        super(Pix2PixTrainer, self).__init__()

    
    def train(self, train_loader, val_loader, writer):

        for self.epoch in range(self.starting_epoch, self.num_epochs):
            
            self.model.set_mode(G='train', D='train')

            i = 0

            for data in train_loader:
                
                real_rgb, real_semseg = \
                    data['rgb'].to(self.device), data['semseg'].to(self.device)

                # Generator forward
                fake_rgb = self.model.G(real_semseg)

                # Update Discriminator
                self.model.set

