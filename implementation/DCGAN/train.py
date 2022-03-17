import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torchsummary import summary

from dcgan import Discriminator, Generator


def save_img(img, title, dir):
    plt.figure(figsize = (4,4))
    plt.imshow(torch.permute(img.cpu(),(1,2,0)))
    plt.axis('off')
    plt.title(title, y=-0.16, fontsize=10)
    plt.savefig(os.path.join(dir, title+'.png'), bbox_inches='tight',pad_inches = 0)
    plt.close()


# Hyper-parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
noise_dim = 100
img_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50
k = 1 
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

# Load dataset
dataset = datasets.MNIST(root="../../datasets/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Initialize model and optimizer
disc = Discriminator(img_dim).to(device)
gen = Generator(noise_dim, img_dim).to(device)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
#summary(disc, (img_dim,), device=device)
#summary(gen, (noise_dim,), device=device)


# The fixed noise to evaluate
fixed_noise = torch.randn((batch_size, noise_dim)).to(device)

# Build result directory
writer = SummaryWriter("logs/")
fake_dir = "images/fake"
if not os.path.exists(fake_dir):
    os.makedirs(fake_dir)
    

# Start training
step = 0
start_time = time.time()
print('Start training: epoch={}, batch_size={}, lr={}, k={}, noise_dim={}, img_dim={}, device={}'\
                .format(num_epochs, batch_size, lr, k, noise_dim, img_dim, device))
for epoch in range(num_epochs):

    # Sample images
    with torch.no_grad():
        fake = gen(fixed_noise)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True, pad_value=1)
        save_img(img_grid_fake, 'Epoch {}'.format(epoch), fake_dir)
        step += 1

    # Train for an epoch
    for batch_id, (real, _) in tqdm(enumerate(loader), ascii=True, desc='Epoch {}/{}'\
                                            .format(epoch+1, num_epochs),total=len(loader)):
        
        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        for _ in range(k):
            z = torch.randn(batch_size, noise_dim).to(device)
            fake = gen(z)
            real = real.to(device)
            disc_real = disc(real).view(-1)
            disc_fake = disc(fake).view(-1)
            loss_real = criterion(disc_real, torch.ones_like(disc_real))
            loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_D = (loss_real + loss_fake) / batch_size
            disc.zero_grad()
            loss_D.backward()
            opt_disc.step()
        
        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        z = torch.randn(batch_size, noise_dim).to(device)
        fake = gen(z)
        disc_fake = disc(fake).view(-1)
        loss_G = criterion(disc_fake, torch.ones_like(disc_fake)) / batch_size
        gen.zero_grad()
        loss_G.backward()
        opt_gen.step()
        
    # Evaluate after one epoch
    print("loss D={:.4f}, loss G={:.4f}, time={:.2f}".format(loss_D,loss_G,time.time()-start_time))
    writer.add_scalar('Loss/Discriminator', loss_D, epoch)
    writer.add_scalar('Loss/Generator', loss_G, epoch)

print('Training finished: time={:.2f}, final loss D={:.4f}, final loss G={:.4f}'\
                    .format(time.time()-start_time, loss_D, loss_G))
