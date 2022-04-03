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

from wgan import Critic, Generator


def save_img(img, title, dir):
    plt.figure(figsize = (4,4))
    plt.imshow(torch.permute(img.cpu(),(1,2,0)))
    plt.axis('off')
    plt.title(title, y=-0.16, fontsize=10)
    plt.savefig(os.path.join(dir, title+'.png'), bbox_inches='tight',pad_inches = 0)
    plt.close()



def grad_penalty(critic, real, fake):

    # Sampling distribution
    coeff = torch.randn(len(real)).view(-1,1,1,1).to(real.device)
    sampled_data = real*coeff + fake*(1-coeff)

    # feed into critic and calculate gradient
    critic_output = critic(sampled_data)
    sampled_grad = torch.autograd.grad(
        outputs=critic_output,
        inputs=sampled_data,
        grad_outputs=torch.ones_like(critic_output),
        # allow gradient from the gradient penalty term to be backproped 
        retain_graph=True, 
        create_graph=True, 
    )[0]

    # calculate gradient penalty term (two-sided-penalty)
    sampled_grad_norm = torch.linalg.vector_norm(sampled_grad, dim=(1,2,3)) # calculate vector 2 norm for each sample in the batch
    sampled_grad_penalty = torch.mean((sampled_grad_norm - 1) ** 2)

    return sampled_grad_penalty, sampled_grad_norm.mean()


# Hyper-parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
penalty_coeff = 10
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
crit = Critic(img_dim).to(device)
gen = Generator(noise_dim, img_dim).to(device)
opt_crit = optim.Adam(crit.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
# summary(crit, (1,28,28), device=device)
# summary(gen, (noise_dim,), device=device)


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
print('Start training: epoch={}, penalty_coeff={}, batch_size={}, lr={}, k={}, noise_dim={}, img_dim={}, device={}'\
                .format(num_epochs, penalty_coeff, batch_size, lr, k, noise_dim, img_dim, device))
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
        
        # Train critic: max D(x)-D(G(z))-grad_penalty = min D(G(z))+grad_penalty-D(x)
        for _ in range(k):
            # calculate critic output and generator output
            z = torch.randn(batch_size, noise_dim).to(device)
            fake = gen(z)
            real = real.to(device)
            crit_real = crit(real).mean()
            crit_fake = crit(fake).mean()
            
            # calculate critic loss and gradient penalty
            gp, grad_norm = grad_penalty(crit, real, fake)
            loss_D = crit_fake + penalty_coeff*gp - crit_real
            loss_D_wo_penalty = crit_fake - crit_real
            crit.zero_grad()
            loss_D.backward()
            opt_crit.step()
        
        # Train Generator: max D(G(z)) = min -D(G(z))
        z = torch.randn(batch_size, noise_dim).to(device)
        fake = gen(z)
        crit_fake = crit(fake)
        loss_G = -1*crit(fake).mean()
        gen.zero_grad()
        loss_G.backward()
        opt_gen.step()
        
        
    # Evaluate after one epoch
    loss_G*=-1
    loss_D_wo_penalty*=-1
    print("loss D={:.4f}, loss G={:.4f}, grad_norm={:.4f}, time={:.2f}".format(loss_D,loss_G,grad_norm,time.time()-start_time))
    writer.add_scalar('Loss/Critic', loss_D, epoch)
    writer.add_scalar('Loss/Generator', loss_G, epoch)
    writer.add_scalar('Critic gradient norm', grad_norm, epoch)

print('Training finished: time={:.2f}, final loss D={:.4f}, final loss G={:.4f}, grad_norm={:.4f}'\
                    .format(time.time()-start_time, loss_D, loss_G, grad_norm))
