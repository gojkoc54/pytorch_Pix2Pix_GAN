import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os 
from PIL import Image


# ========================================================================================
# === DATA HANDLING ===
# ========================================================================================


class Pix2PixDataset(Dataset):

    def __init__(self, root_folder, in_folder, out_folder, transform=None):
        super(Pix2PixDataset, self).__init__()

        self.root_folder = os.path.abspath(root_folder)
        self.in_folder = os.path.join(root_folder, in_folder)
        self.out_folder = os.path.join(root_folder, out_folder)

        self.in_img_names = os.listdir(self.in_folder)
        self.in_img_names.sort()

        self.transform = transform

        self.len = len(self.in_img_names)
    

    def __len__(self):

        return self.len


    def __getitem__(self, idx):
        
        img_name = self.in_img_names[idx]
        in_img_path = os.path.join(self.in_folder, img_name)
        out_img_path = os.path.join(self.out_folder, img_name)

        in_img = Image.open(in_img_path).convert('RGB')   
        out_img = Image.open(out_img_path).convert('RGB')

        if self.transform is None:
            self.transform = transforms.ToTensor()

        in_img = self.transform(in_img)
        out_img = self.transform(out_img)

        return {'in': in_img, 'out': out_img}




def load_datasets_UNet(
    train_filepath, val_filepath, crop_size, new_size, batch_size, 
    num_workers=0, shuffle_train=True, shuffle_val=False
    ):
    
    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.Resize(new_size), 
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])


    # Load the training set from the root folder:
    train_dataset = ImageFolder(root = train_filepath, transform = transform) if train_filepath is not None else []

    # Load the validation set from the root folder:
    val_dataset = ImageFolder(root = val_filepath, transform = transform) if val_filepath is not None else []

    # Create DataLoader objects for both the training and validation sets:  
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers) 
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers=num_workers)

    return train_loader, val_loader



def load_dataset_Pix2Pix(
    train_filepath, val_filepath, crop_size, new_size, batch_size, in_folder, out_folder,
    num_workers=0, shuffle_train=True, shuffle_val=False
    ):

    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(new_size),
        transforms.ToTensor()
    ])

    train_dataset = Pix2PixDataset(
        root_folder=train_filepath, in_folder=in_folder, out_folder=out_folder, transform=transform
        ) if train_filepath is not None else []
    
    val_dataset = Pix2PixDataset(
        root_folder=val_filepath, in_folder=in_folder, out_folder=out_folder, transform=transform
        ) if val_filepath is not None else []

    # Create DataLoader objects for both the training and validation sets:  
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle_train, num_workers=num_workers) 
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = shuffle_val, num_workers=num_workers)

    return train_loader, val_loader
