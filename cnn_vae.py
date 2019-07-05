import numpy as np
import pandas as pd
import gc
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import os.path as osp
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data as D
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_ssim
import time


path_x = 'C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\\Denoise for inpainting\\Code\\occulted_output\\20_percent\\100\\100'
path_y = 'C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\\Denoise for inpainting\\Code\\groundtruth_output\\resized_100\\100'

class Data(D.Dataset):
    """Creates a data-set"""

    def __init__(self,root_x, root_y):
        """Initialize the dataset"""
        self.filenames_x = []
        self.filenames_y = []
        self.root_x = root_x
        self.root_y = root_y
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        filenames_x = glob.glob(osp.join(path_x, '*.jpeg'))
        filenames_y = glob.glob(osp.join(path_y, '*.jpeg'))
        for (x,y) in zip(filenames_x, filenames_y):
            self.filenames_x.append(x)
            self.filenames_y.append(y)
        self.len = len(self.filenames_x)

    def __getitem__(self, index):
        """Allowing slicing in the data-set"""
        image_x = Image.open(self.filenames_x[index])
        image_y = Image.open(self.filenames_y[index])
        image_y = image_y.convert('LA')
        return self.transform(image_x), self.transform(image_y)

    def __len__(self):
        """returning the length of the data-set"""
        return self.len


imgs = Data(path_x, path_y)


# A function to show an image
def imshow(img):
    trans = transforms.ToPILImage()(img)
    trans.show()
# Checking that I acutally have the pics associated
i = np.random.randint(0,len(imgs))
sample = imgs[i]
x, y = sample

# imshow(x)
# imshow(y)

data_loader = D.DataLoader(imgs, batch_size=128, shuffle=True, num_workers=0)


class Net(nn.Module):
    def __init__(self, num_latent):
        super().__init__()
        
        #So here we will first define layers for encoder network
        self.encoder = nn.Sequential(nn.Conv2d(1, 50, 3, padding=1),    # 100*100*30
                                     nn.MaxPool2d(2, 2),                # 50*50*30
                                     nn.BatchNorm2d(50),
                                     nn.ReLU(True),
                                     nn.Conv2d(50, 100, 3, padding=1),   # 50*50*50
                                     nn.MaxPool2d(2, 2),                # 25*25*50
                                     nn.BatchNorm2d(100),
                                     nn.ReLU(True),
                                     nn.Conv2d(100, 200, 3, padding=1),   # 25*25*75
                                     nn.ReLU(True),
                                     nn.Conv2d(200, 350, 4, padding=1),  # 24*24*75
                                     nn.MaxPool2d(2, 2),                # 12*12*100
                                     nn.Conv2d(350, 500, 3),            # 10*10*200
                                     nn.MaxPool2d(2, 2))                # 5*5*200


        #These two layers are for getting logvar and mean
        self.fc1 = nn.Linear(12500, 6000)
        # self.fc2 = nn.Linear(6000, 3000)
        self.mean = nn.Linear(6000, num_latent)
        self.var = nn.Linear(6000, num_latent)
        
        #######The decoder part
        #This is the first layer for the decoder part
        self.expand = nn.Linear(num_latent, 6000)
        # self.fc3 = nn.Linear(3000, 6000)
        self.fc4 = nn.Linear(6000, 12500) # this represents a 5*5*200 cube
        self.decoder = nn.Sequential(nn.ConvTranspose2d(500, 200, 3, stride=2), # 11*11*100
                                     nn.BatchNorm2d(200),
                                     nn.ConvTranspose2d(200, 100, 3, stride=2), # 23*23*50
                                     nn.BatchNorm2d(100),
                                     nn.ConvTranspose2d(100, 35, 6, stride=2),  # 50*50*25
                                     nn.BatchNorm2d(35),
                                     nn.ConvTranspose2d(35, 1, 2, stride=2))  # 100*100*1

        
    def enc_func(self, x):
        #here we will be returning the logvar(log variance) and mean of our network
        x = self.encoder(x)
        # print(x.shape)
        x = x.view([-1, 12500])
        x = self.fc1(x)
        # x = self.fc2(x)
        
        mean = self.mean(x)
        logvar = self.var(x)
        return mean, logvar
    
    def dec_func(self, z):
        #here z is the latent variable state
        z = self.expand(z)
        # z = F.dropout2d(self.fc3(z), 0.2)
        z = self.fc4(z)
        z = z.view([-1, 500, 5, 5])
        
        out = self.decoder(z)
        out = torch.sigmoid(out)
        return out
    
    def get_hidden(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)   # get std
            noise = torch.randn_like(mean)   # get the noise of standard distribution
            return noise.mul(std).add_(mean)
        else:
            return mean
    
    def forward(self, x):
        mean, logvar = self.enc_func(x)
        z = self.get_hidden(mean, logvar)
        out = self.dec_func(z)
        # print(out.shape)
        return out, mean, logvar

def VAE_loss(x_recon, y, mean, logvar):
    base_loss = nn.BCEWithLogitsLoss()
    ### Trying to solve the error with BCE loss 
    # y = torch.sigmoid(y)   
    # x_recon = torch.sigmoid(x_recon)
    ###########################################
    # eps = 1e-8
    bce_loss = base_loss(x_recon, y)
    # Scale the following losses with this factor
    scaling_factor = x_recon.shape[0]*x_recon.shape[1]*x_recon.shape[2]*x_recon.shape[3]
    
    ####Now we are gonna define the KL divergence loss
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.05 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
    kl_loss /= scaling_factor    # trying without this values
    
    return bce_loss + kl_loss

def train(trainloader, epochs, model, device, optimizer):
    for epoch in range(epochs):
        start = time.time()
        model.train()
        model.to(device)
        train_loss = 0
        for i,(images, target) in enumerate(trainloader):
            images = images.to(device)
            target = target.to(device)
           
            optimizer.zero_grad()
            out, mean, logvar = model(images)
            out = out.to(device)
            
            ### VAE loss
            # loss = VAE_loss(out, images, mean, logvar)
            
            ### MSE loss
            criterion = nn.MSELoss()
            loss = criterion(out, images)

            # Backpropagation and optimization
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            
            # if i % 100 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, i * len(data_loader), 
            #         len(data_loader.dataset), 100. * i / len(data_loader), loss.item()))
        
        end = time.time()  
        elasped_time = (end - start)/60          
        print('========================> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / i))
        print("========================> This epoch took {:4f} mins to be completed".format(elasped_time))
######Setting all the hyperparameters
epochs = 50
num_latent = 1500

model = Net(num_latent)
device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.00001)


train(data_loader, epochs, model, device, optimizer)



##################################
############ Testing #############
##################################

# Create a directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Getting a never seen image 
test_x = glob.glob(osp.join(path_x, '*.jpg'))

for j in range (0,10):
    i = np.random.randint(0,len(test_x))
    test_image = Image.open(test_x[i])

    # Try reconstructing on test data
    with torch.no_grad():
        test_image = torchvision.transforms.ToTensor()(test_image)
        test_image = test_image.view([1, 1, 100, 100]).to(device)
        test_reconst, mean, logvar = model(test_image)

        torchvision.utils.save_image(test_image.data, os.path.join(sample_dir, 'original-{}-{}_mse.png'.format(epochs+1, j)))
        torchvision.utils.save_image(test_reconst.data, os.path.join(sample_dir, 'reconst-{}-{}_mse.png'.format(epochs+1, j)))
