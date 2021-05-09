import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


# ========================================================================================
# === DATA LOADING ===
# ========================================================================================

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
