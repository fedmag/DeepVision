
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

data_loader = D.DataLoader(imgs, batch_size=256, shuffle=True, num_workers=0)


class Net(nn.Module):
    def __init__(self, num_latent):
        super().__init__()
        
        #So here we will first define layers for encoder network (VGG_16)
        self.encoder = nn.Sequential(nn.Conv2d(1, 64, 3),    # 98*98*64
                                     nn.Conv2d(64, 64, 3),   # 96*96*64   
                                     nn.MaxPool2d(2, 2),     # 48*48*64
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(64, 128, 3),    # 46*46*128
                                     nn.Conv2d(128, 128, 3),   # 44*44*128
                                     nn.MaxPool2d(2, 2),       # 22*22*128
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(128, 256, 3),   # 20*20*256
                                     nn.Conv2d(256, 256, 3),   # 18*18*256
                                     nn.Conv2d(256, 256, 3),   # 16*16*256
                                     nn.MaxPool2d(2, 2),       # 8*8*256
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(256, 512, 3),   # 6*6*512
                                     nn.Conv2d(512, 512, 3),   # 4*4*512
                                     nn.BatchNorm2d(512))
                                    


        #These two layers are for getting logvar and mean
        self.fc1 = nn.Linear(8192, 4096)
        # self.fc2 = nn.Linear(6000, 3000)
        self.mean = nn.Linear(4096, num_latent)
        self.var = nn.Linear(4096, num_latent)
        
        #######The decoder part
        #This is the first layer for the decoder part
        self.expand = nn.Linear(num_latent, 4096)
        # self.fc3 = nn.Linear(3000, 6000)
        self.fc4 = nn.Linear(4096, 8192) # this represents a 4*4*512 cube
        self.decoder = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2), # 10*10*256
                                     nn.BatchNorm2d(256),
                                     nn.LeakyReLU(True),
                                     nn.ConvTranspose2d(256, 128, 5, stride=2), # 23*23*128
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(True),
                                     nn.ConvTranspose2d(128, 64, 4, stride=2),  # 48*48*64
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(True),
                                     nn.ConvTranspose2d(64, 1, 6, stride=2))    # 100*100*1
                                     

        
    def enc_func(self, x):
        #here we will be returning the logvar(log variance) and mean of our network
        x = self.encoder(x)
        # print(x.shape)
        x = x.view([-1, 8192])
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
        z = z.view([-1, 512, 4, 4])
        
        out = self.decoder(z)
        out = torch.sigmoid(out)
        # print(out.shape)
        return out
    
    def get_hidden(self, mean, logvar):
        std = torch.exp(0.5*logvar)   # get std
        noise = torch.randn_like(mean)   # get the noise of standard distribution
        return noise.mul(std).add_(mean)

    
    def forward(self, x):
        mean, logvar = self.enc_func(x)
        z = self.get_hidden(mean, logvar)
        out = self.dec_func(z)
        # print(out.shape)
        return out, mean, logvar

def VAE_loss(x_recon, y, mean, logvar):
    
    base_loss = nn.MSELoss()

    ###########################################
    mse_loss = base_loss(x_recon, y)
    # Scale the following losses with this factor
    scaling_factor = x_recon.shape[0]*x_recon.shape[1]*x_recon.shape[2]*x_recon.shape[3]
    
    ####Now we are gonna define the KL divergence loss
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.005 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
    kl_loss /= scaling_factor    # trying without this values
    
    return mse_loss + kl_loss

def train(trainloader, epochs, model, device, optimizer):
    avg_losses = []
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
            loss = VAE_loss(out, target, mean, logvar)

            # Backpropagation and optimization
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        ### Statistics   
        avg_losses.append(train_loss/len(trainloader))
        end = time.time()  
        elasped_time = (end - start)/60          
        print('========================> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader)))
        print("========================> This epoch took {:4f} mins to be completed".format(elasped_time))
        
        # Printing images
        with torch.no_grad():
            sample_dir = 'samples'
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            test_image = Image.open('C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\\Denoise for inpainting\\Code\\occulted_output\\20_percent\\100\\100\\ea34b00f2ff5033ed1584d05fb1d4e9fe777ead218ac104497f5c978a7ebb0bb_640.jpg')
            test_image = torchvision.transforms.ToTensor()(test_image)
            test_image = test_image.view([1, 1, 100, 100]).to(device)
            test_reconst, _ , X = model(test_image)
            torchvision.utils.save_image(test_reconst.data, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch)))
    
    # Plotting the loss function
    plt.plot(avg_losses)


######Setting all the hyperparameters
epochs = 45
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

# Create a directory does not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)



# # Getting a never seen image 
# test_x = glob.glob(osp.join(path_x, '*.jpg'))

# for j in range (0,10):
#     i = np.random.randint(0,len(test_x))
#     test_image = Image.open(test_x[i])

#     # Try reconstructing on test data
#     with torch.no_grad():
#         test_image = torchvision.transforms.ToTensor()(test_image)
#         test_image = test_image.view([1, 1, 100, 100]).to(device)
#         test_reconst, mean, logvar = model(test_image)

#         torchvision.utils.save_image(test_image.data, os.path.join(sample_dir, 'original-{}-{}_mse.png'.format(epochs+1, j)))
#         torchvision.utils.save_image(test_reconst.data, os.path.join(sample_dir, 'reconst-{}-{}_mse.png'.format(epochs+1, j)))

######### SIMPLE 
# Try reconstructing on test data

test_image = Image.open('C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\\Denoise for inpainting\\Code\\occulted_output\\20_percent\\100\\100\\ea34b00f2ff5033ed1584d05fb1d4e9fe777ead218ac104497f5c978a7ebb0bb_640.jpg')
test_image = torchvision.transforms.ToTensor()(test_image)
test_image = test_image.view([1, 1, 100, 100]).to(device)
test_reconst, _ , X = model(test_image)

torchvision.utils.save_image(test_image.data, os.path.join(sample_dir, 'original-{}.png'.format(epochs)))
torchvision.utils.save_image(test_reconst.data, os.path.join(sample_dir, 'reconst-{}.png'.format(epochs)))
