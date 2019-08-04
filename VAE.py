import numpy as np
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
import time


path_x = 'C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\\Denoise for inpainting\\Code FACES\\occulted_output\\40_percent\\100\\' # path occulted input
path_y = 'C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\\Denoise for inpainting\\Code FACES\\groundtruth_output\\resized_100\\' # path ground truth


# Creating a dataset and then a data loader
class Data(D.Dataset):
    """Creates a data-set"""

    def __init__(self,root_x, root_y):
        """Initialize the dataset"""
        self.filenames_x = []
        self.filenames_y = []
        self.root_x = root_x
        self.root_y = root_y
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        filenames_x = glob.glob(osp.join(path_x, '*.png'))
        filenames_y = glob.glob(osp.join(path_y, '*.png'))
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

data_loader = D.DataLoader(imgs, batch_size=64, shuffle=True, num_workers=0)


################################################################
###################### MODEL ###################################
################################################################
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
                                     nn.LeakyReLU(True))

        #These two layers are for getting logvar and mean
        self.fc1 = nn.Linear(16384, 8000)
        self.mean = nn.Linear(8000, num_latent)
        self.var = nn.Linear(8000, num_latent)
        
        #######The decoder part
        #This is the first layer for the decoder part
        self.expand = nn.Linear(num_latent, 8000)
        self.fc4 = nn.Linear(8000, 16384) # this represents a 8*8*256 cube
        self.decoder = nn.Sequential(nn.ConvTranspose2d(256, 128, 5, stride=2), # 19*19*128
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(True),
                                     nn.ConvTranspose2d(128, 64, 5, stride=2), # 41*41*64
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(True),
                                     nn.ConvTranspose2d(64, 32, 5, stride=2),  # 85*85*32
                                     nn.BatchNorm2d(32),
                                     nn.LeakyReLU(True),
                                     nn.ConvTranspose2d(32, 16, 6, stride=1),    # 89*89*16
                                     nn.BatchNorm2d(16),
                                     nn.LeakyReLU(True),
                                     nn.ConvTranspose2d(16, 8, 6, stride=1), # 94*94*8
                                     nn.BatchNorm2d(8),
                                     nn.LeakyReLU(True),
                                     nn.ConvTranspose2d(8, 1, 6, stride=1)) # 100*100*1
                                     

        
    def enc_func(self, x):
        #here we will be returning the logvar(log variance) and mean of our network
        x = self.encoder(x)
        x = x.view([-1, 16384])
        x = self.fc1(x)
        
        mean = self.mean(x)
        logvar = self.var(x)
        return mean, logvar
    
    def dec_func(self, z):
        #here z is the latent variable state
        z = self.expand(z)
        z = self.fc4(z)
        z = z.view([-1, 256, 8, 8])
        out = self.decoder(z)
        out = torch.sigmoid(out)

        return out
    
    def get_hidden(self, mean, logvar):
        std = torch.exp(0.5*logvar)   # get std
        noise = torch.randn_like(mean)   # get the noise of standard distribution
        return noise.mul(std).add_(mean)

    
    def forward(self, x):
        mean, logvar = self.enc_func(x)
        z = self.get_hidden(mean, logvar)
        out = self.dec_func(z)

        return out, mean, logvar

def VAE_loss(x_recon, y, mean, logvar):
    ### MSE
    base_loss = nn.MSELoss()
    loss = base_loss(x_recon, y)

    # Scale the following losses with this factor
    scaling_factor = x_recon.shape[0]*x_recon.shape[1]*x_recon.shape[2]*x_recon.shape[3]
    
    ####Now we are gonna define the KL divergence loss
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.005 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
    kl_loss /= scaling_factor 
    
    return loss + kl_loss

def train(trainloader, start_epochs, epochs, model, device, optimizer, avg_losses):
    if len(avg_losses) > 1:
        avg_losses = avg_losses
    else:
        avg_losses = []
    ### Training
    for epoch in range(start_epochs+1, epochs+1):
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
            
            # VAE loss
            loss = VAE_loss(out, target, mean, logvar)

            # Backpropagation and optimization
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        ### Statistics   
        avg_losses.append(train_loss/len(trainloader))
        end = time.time()  
        elasped_time = (end - start)/60          
        print('=======> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss / len(data_loader)))
        print("=======> This epoch took {:.3f} mins to be completed".format(elasped_time))
        
        # Printing images
        with torch.no_grad():
            sample_dir = 'samples40'
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            test_image = Image.open('C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\\Denoise for inpainting\\Code FACES\\occulted_output\\40_percent\\100\\00246.png') # This is a random picture, just to check the progress
            test_image = torchvision.transforms.ToTensor()(test_image)
            test_image = test_image.view([1, 1, 100, 100]).to(device)
            test_reconst, _ , X = model(test_image)
            torchvision.utils.save_image(test_reconst.data, os.path.join(sample_dir, 'reconst-{}-changing.png'.format(epoch)))
            if epoch % 20 == 0:
                test_dir = 'test40'
                if not os.path.exists(test_dir):
                    os.makedirs(test_dir)
                # Getting a never seen image 
                path_test_x = 'C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\\Denoise for inpainting\\Code FACES\\occulted_output\\40_percent\\test\\'
                test_x = glob.glob(osp.join(path_test_x, '*.png'))

                for j in range (0,10):
                    i = np.random.randint(0,len(test_x))
                    test_image = Image.open(test_x[i])

                    # Try reconstructing on test data
                    with torch.no_grad():
                        test_image = torchvision.transforms.ToTensor()(test_image)
                        test_image = test_image.view([1, 1, 100, 100]).to(device)
                        test_reconst, mean, logvar = model(test_image)

                        torchvision.utils.save_image(test_image.data, os.path.join(test_dir, '{}-original_test.png'.format(j)))
                        torchvision.utils.save_image(test_reconst.data, os.path.join(test_dir, '{}-reconst_test.png'.format(j)))

    # Plotting the loss function
    plt.plot(avg_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    ### Saving the model
    torch.save({
            'epochs': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'average losses' : avg_losses
            }, 'checkpoint-{}-1500-changing.pth.tar'.format(epochs))



######Setting all the hyperparameters
start_epochs = 0
epochs = 50
num_latent = 1500
avg_losses = []

model = Net(num_latent)
device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train(data_loader,start_epochs, epochs, model, device, optimizer, avg_losses)

####################################### TEST ######################################
# Getting a never seen images, TESTING THE MODEL
test_path_x = 'C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\\Denoise for inpainting\\Code FACES\\occulted_output\\40_percent\\test' # path test images
test_x = glob.glob(osp.join(test_path_x, '*.png'))

sample_dir = '40_occulted_test'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
for i in range (0,len(test_x)):
    test_image = Image.open(test_x[i])
    # Try reconstructing on test data
    with torch.no_grad():
        test_image = test_image.convert('L')
        test_image = torchvision.transforms.ToTensor()(test_image)
        test_image = test_image.view([-1, 1, 100, 100]).to(device)
        test_reconst, mean, logvar = model(test_image)
        torchvision.utils.save_image(test_reconst.data, os.path.join(sample_dir, '{}-reconst-.png'.format(i)))
        torchvision.utils.save_image(test_image.data, os.path.join(sample_dir, '{}-original.png'.format(i)))

######################################### PAUSE AND/OR LOAD THE MODEL ######################################
## Loading the model
checkpoint = torch.load('checkpoint-50-1500-changing.pth.tar') # Insert number of epochs of the trained model instead of '50'
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Do not load this in order to use the trained model and decrease the size of the loading
start_epochs = checkpoint['epochs']
avg_losses = checkpoint['average losses']

### Resume the training
# epochs = 200 # epochs you want to reach after restarting the training
# train(data_loader,start_epochs, epochs, model, device, optimizer, avg_losses)

### Getting a file with the losses
with open('losses_faces_40.txt', 'w') as f:
    for i, loss in enumerate(avg_losses):
        data = (i, loss)
        f.write(str(data))
