import torch, torchvision
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
import torch.nn.functional as F

#### Data
dir1 = 'C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\Denoise for inpainting\\Code\\occulted_output\\20_percent\\100\\'

dir2 = 'C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\Denoise for inpainting\\Code\\groundtruth_output\\resized_100\\'



# Hyper-parameters
image_size = 10000 # 100x100
h_dim = 350
z_dim = 20
num_epochs = 3
batch_size = 1
learning_rate = 0.001

#### Data
data_path = dir1
test_path = dir2

transform  = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=3,shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(root=test_path,transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,num_workers=3,shuffle=False)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create a directory if does not exist
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=10000, h_dim=700, z_dim=70):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim,h_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(h_dim, z_dim)
        self.fc5 = nn.Linear(z_dim, h_dim)
        self.fc6 = nn.Linear(h_dim, h_dim)
        self.fc7 = nn.Linear(h_dim, image_size)
        
    def encode(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x), self.fc4(x)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc5(z)
        z = self.fc6(z)
        z = F.relu(z)
        z = self.fc7(z)
        return torch.sigmoid(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var




model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start training
for epoch in tqdm(range(num_epochs)):
    for i, (x,y) in enumerate(zip(train_loader, test_loader),0):
        # Forward pass
        x = x[0]
        x = x.view(-1, image_size)
        x_reconst, mu, log_var = model(x)
        
        y = y[0]
        y = y.view(-1,image_size)
        # Compute reconstruction loss and kl divergence
        reconst_loss = F.binary_cross_entropy(x_reconst, y)
        kl_div = - 0.001 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 250 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Loss {}, Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, num_epochs, i+1, len(train_loader),loss.item(), reconst_loss.item(), kl_div.item()))

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images = data[0]
            images = images.view(-1, image_size)
            recon_image, s, mu = model(images)
            recon_image_ = recon_image.view(-1, 1, 100, 100)
            if i % 100 == 0:
                save_image(recon_image_, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))
