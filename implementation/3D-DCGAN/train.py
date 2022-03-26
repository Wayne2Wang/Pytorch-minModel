import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torchsummary import summary

from dataset import *
from dcgan_3d import Discriminator, Generator


# Hyper-parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
cr_lambda = 1 #10
d_acc_thresh = 0.8
lr_D = 1e-4
lr_G = 0.0025
batch_size = 16
batch_norm = True
num_epochs = 200
k = 2
noise_dim = 400
img_c, img_d, img_h ,img_w = 1, 64, 64, 64
disp_num = 8
save_every = 1


def hyperparameter_str():
    return 'epoch={}, batch_size={}, batch_norm={}, cr_lambda={}, d_acc_thresh={}, lr_D={}, lr_G={}, k={}, noise_dim={}, img_dim={}, save_every={}, device={}'\
                .format(num_epochs, batch_size, batch_norm, cr_lambda, d_acc_thresh, lr_D, lr_G, k, noise_dim, (img_c, img_d, img_h ,img_w), save_every, device)


def save_img(img, title, dir, disp_num):

    # save png
    SavePloat_Voxels(img, dir, title, disp_num)


# Load dataset
dataset = ShapeNetDataset()
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


# Initialize model and optimizer
disc = Discriminator(img_c, batch_norm=batch_norm).to(device)
gen = Generator(noise_dim, img_c, batch_norm=batch_norm).to(device)
opt_disc = optim.Adam(disc.parameters(), lr=lr_D)
opt_gen = optim.Adam(gen.parameters(), lr=lr_G)
criterion = nn.BCEWithLogitsLoss() # better stability than nn.BCELoss()
criterion_cr = nn.MSELoss()

# Uncomment to view model details
#summary(disc, (1,img_d, img_h ,img_w), device=device)
#summary(gen, (noise_dim,), device=device)


# The fixed noise to evaluate
fixed_noise = torch.randn((disp_num, noise_dim)).to(device)

# Build result directory
writer = SummaryWriter("logs/")
fake_dir = "images/fake"
if not os.path.exists(fake_dir):
    os.makedirs(fake_dir)
    

# Start training
start_time = time.time()
print('Start training: '+ hyperparameter_str())
for epoch in range(num_epochs):

    # Sample images
    if epoch % save_every == 0:
        with torch.no_grad():
            gen.eval()
            fake = gen(fixed_noise) # generate the fake 5D image
            fake = fake.squeeze(1).cpu().numpy() # squeeze back to 3D numpy array and move to cpu for plotting
            save_img(fake, 'Epoch_{}'.format(epoch), fake_dir, disp_num)
            gen.train()

    # Train for an epoch
    for batch_id, (real, reg_real) in tqdm(enumerate(loader), ascii=True, desc='Epoch {}/{}'\
                                            .format(epoch+1, num_epochs),total=len(loader)):
        
        if batch_id == 0 and epoch == 0:
            save_img(real[:disp_num].squeeze(1).cpu().numpy(), 'real_{}'.format(epoch), fake_dir, disp_num)
            save_img(reg_real[:disp_num].squeeze(1).cpu().numpy(), 'reg_real_{}'.format(epoch), fake_dir, disp_num)
        
        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        for _ in range(k):
            
            # generate sample and feed to discriminator
            z = torch.randn(batch_size, noise_dim).to(device)
            real = real.to(device)
            fake = gen(z)
            pre_act_disc_real, disc_real = disc(real)
            _, disc_fake = disc(fake)
            disc_real, disc_fake = disc_real.view(-1), disc_fake.view(-1)

            # if discriminator acc is below threshold, calculate discriminator loss
            d_acc_real = torch.sum(torch.sigmoid(disc_real)>0.5) / disc_real.shape[0]
            d_acc_fake = torch.sum(torch.sigmoid(disc_fake)<0.5) / disc_real.shape[0]
            d_acc = (d_acc_real + d_acc_fake) / 2
            if d_acc < d_acc_thresh:
                loss_real = criterion(disc_real, torch.ones_like(disc_real))
                loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                loss_D = (loss_real + loss_fake) 
            else:
                loss_D = 0

            # CR loss
            reg_real = reg_real.to(device)
            pre_act_disc_reg_real, _ = disc(reg_real)
            loss_cr = criterion_cr(pre_act_disc_reg_real, pre_act_disc_real)

            # sum of both loss
            loss_D = (loss_D + cr_lambda*loss_cr) / batch_size

            # backward and optimize
            disc.zero_grad()
            loss_D.backward()
            opt_disc.step()
        
        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        z = torch.randn(batch_size, noise_dim).to(device)
        fake = gen(z)
        _, disc_fake = disc(fake)
        disc_fake = disc_fake.view(-1)
        loss_G = criterion(disc_fake, torch.ones_like(disc_fake)) / batch_size
        
        gen.zero_grad()
        loss_G.backward()
        opt_gen.step()
        
    # save model
    torch.save(gen, fake_dir+'/generator.pt'.format(epoch))
    torch.save(disc, fake_dir+'/discriminator.pt'.format(epoch))

    # Evaluate after one epoch
    print("loss D={:.4f}, loss G={:.4f}, time={:.2f}".format(loss_D,loss_G,time.time()-start_time))
    writer.add_scalar('Loss/Discriminator', loss_D, epoch)
    writer.add_scalar('Loss/Generator', loss_G, epoch)

print('Training finished: time={:.2f}, final loss D={:.4f}, final loss G={:.4f}'\
                    .format(time.time()-start_time, loss_D, loss_G))
