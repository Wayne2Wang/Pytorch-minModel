import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision.transforms import InterpolationMode
from torch.nn.functional import one_hot

from unet import Unet

PALETTE = [(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),(  0,  0,  0),
           (111, 74,  0),( 81,  0, 81),(128, 64,128),(244, 35,232),(250,170,160),
           (230,150,140),( 70, 70, 70),(102,102,156),(190,153,153),(180,165,180),
           (150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30),
           (220,220,  0),(107,142, 35),(152,251,152),( 70,130,180),(220, 20, 60),
           (255,  0,  0),(  0,  0,142),(  0,  0, 70),(  0, 60,100),(  0,  0, 90),
           (  0,  0,110),(  0, 80,100),(  0,  0,230),(119, 11, 32),(  0,  0,142)]

def save_img(img, title, dir):
    plt.figure(figsize = (6,6))
    plt.imshow(torch.permute(img,(1,2,0)))
    plt.axis('off')
    plt.title(title, y=-0.16, fontsize=10)
    plt.savefig(os.path.join(dir, title+'.png'), bbox_inches='tight',pad_inches = 0)
    plt.close()

def pred2color(pred):
    pred = torch.argmax(pred, dim=0)
    color = torch.zeros(3, pred.shape[0], pred.shape[1])
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            color[:,i,j]=torch.tensor(PALETTE[pred[i,j]])
    return color/255



# Hyper-parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
batch_size = 2
num_epochs = 50
img_height, img_width, img_channels = 1024//2, 2048//2, 3
label_height, label_width, label_dim = 324, 836, 35
img_dim = img_height*img_width*img_channels
img_dim_str = '({},{},{})'.format(img_height, img_width, img_channels)
label_dim_str = '({},{},{})'.format(label_height, label_width, label_dim)



# Load dataset; img size = 2048*1024*3
i_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_height,img_width), interpolation=InterpolationMode.BILINEAR)
])
t_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((label_height,label_width), interpolation=InterpolationMode.NEAREST)
])
dataset = datasets.Cityscapes(root="../../datasets/Cityscapes/", transform=i_trans, target_transform=t_trans,\
                                split='train',mode='fine', target_type='semantic')
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


# Initialize model and optimizer
model = Unet(img_channels, label_dim).to(device)
#summary(model, (3,1024,512), device=device)
opt = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# The image to save after each epoch
sample_img = dataset[0][0].unsqueeze(0).to(device)

# Build result directory
writer = SummaryWriter("logs/")
save_img_dir = "images/sample"
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)
path_to_model = 'logs\\checkpoints\\'
if not os.path.exists(path_to_model):
    os.makedirs(path_to_model)

# Start training
start_time = time.time()
print('Start training: epoch={}, batch_size={}, lr={}, img_dim={}, label_dim={}, device={}'\
                .format(num_epochs, batch_size, lr, img_dim_str, label_dim_str, device))
for epoch in range(num_epochs):

    # Sample images
    with torch.no_grad():
        sample_label = model(sample_img).squeeze(0).cpu()
        save_img(pred2color(sample_label), 'Epoch {}'.format(epoch), save_img_dir)
        
    # Train for an epoch
    for batch_id, (img, label) in tqdm(enumerate(loader), ascii=True, desc='Epoch {}/{}'\
                                            .format(epoch+1, num_epochs),total=len(loader)):
        # Move to GPU if available
        img = img.to(device)
        label = (label*255).to(torch.long).squeeze(1).to(device)

        # Prediction
        pred = model(img)

        # Calculate loss and backprop
        loss = criterion(pred, label)
        model.zero_grad()
        loss.backward()
        opt.step()
    
    # Evaluate after one epoch
    print("loss={:.4f}, time={:.2f}".format(loss,time.time()-start_time))
    writer.add_scalar('Loss/train', loss, epoch)

    # Save model for every epoch
    torch.save({'model_state_dict': model.state_dict(), 
                'epochs': epoch, 
                'loss':loss.item(), 
                'img_dim': img_dim_str, 
                'label_dim': label_dim_str}, path_to_model+'unet_{}.pt'.format(epoch))

print('Training finished: time={:.2f}, final loss={:.4f}'\
                    .format(time.time()-start_time, loss))


