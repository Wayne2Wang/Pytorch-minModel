import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.d = nn.Sequential(
            # input image shape: 28x28
            nn.Conv2d(1,64, kernel_size=2, stride=2), # (28-2)/2+1=14
            nn.LeakyReLU(0.2),
            nn.LayerNorm([64,14,14]),
            #nn.BatchNorm2d(64),
            nn.Conv2d(64,128, kernel_size=4, stride=2), # (14-4)/2+1=6
            nn.LeakyReLU(0.2),
            nn.LayerNorm([128,6,6]),
            #nn.BatchNorm2d(128),
            nn.Conv2d(128,256, kernel_size=2, stride=2), # (6-2)/2+1=3
            nn.LeakyReLU(0.2),
            nn.LayerNorm([256,3,3]),
            #nn.BatchNorm2d(256),
            nn.Conv2d(256,1, kernel_size=3, stride=1), # 1
        )
        
    def forward(self, x):
        out = self.d(x)
        return out
    
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.fc1 = nn.Linear(noise_dim, 3*3*512)
        self.bn1 = nn.BatchNorm2d(512)
        self.trans_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2) # 7x7
        self.bn2 = nn.BatchNorm2d(256)
        self.trans_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 14x14
        self.bn3 = nn.BatchNorm2d(128)
        self.trans_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 28x28
        self.bn4 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64,1, kernel_size=1, stride=1) # 28x28

        # activations
        self.relu = nn.ReLU()
        self.tanh  = nn.Tanh()
    
    def forward(self, z):
        z = self.fc1(z)
        z = z.reshape(-1,512,3,3)
        z = self.relu(self.trans_conv1(self.bn1(z)))
        z = self.relu(self.trans_conv2(self.bn2(z)))
        z = self.relu(self.trans_conv3(self.bn3(z)))
        z = self.tanh(self.conv1(self.bn4(z)))
        return z