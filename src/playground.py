import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from models import *
from utils import *
from trainer import *

root_folder = "/home/gojkoc54/GOJKO/Faks/KV/data/pix2pix"
crop_size = (256, 256)
new_size = (256, 256)
batch_size = 1
rgb_folder = "rgb_images"
semseg_folder = "sem_seg_images"

loader, _ = load_dataset_Pix2Pix(
    root_folder, None, crop_size, new_size, batch_size, rgb_folder, semseg_folder
    )

P2P_params = {}
P2P_params['G_in_channels'] = 3 
P2P_params['G_hidden_channels'] = 64
P2P_params['G_out_channels'] = 3
P2P_params['G_depth'] = 4
P2P_params['D_in_channels'] = 6
P2P_params['D_hidden_channels'] = 64
P2P_params['D_depth'] = 3

n_gpu = torch.cuda.device_count()
device = torch.device("cuda:0" if n_gpu else "cpu")

devices = {'D' : device, 'G' : device}
model = Pix2Pix(P2P_params).to(devices)

G_optimizer = torch.optim.Adam(model.G.parameters())
D_optimizer = torch.optim.Adam(model.D.parameters())
GAN_criterion = nn.BCEWithLogitsLoss()
L1_criterion = nn.L1Loss()
lambda_L1 = 0.5

trainer_params = {}
trainer_params['model'] = model
trainer_params['G_optimizer'] = G_optimizer
trainer_params['D_optimizer'] = D_optimizer
trainer_params['GAN_criterion'] = GAN_criterion
trainer_params['L1_criterion'] = L1_criterion
trainer_params['lambda_L1'] = lambda_L1
trainer_params['batch_size'] = batch_size
trainer_params['device'] = device
trainer_params['starting_epoch'] = 0
trainer_params['num_epochs'] = 10

trainer = Pix2PixTrainer(trainer_params)

trainer.train(loader, None, None)


if __name__ == '__main__':

    a = 1
    print(a)