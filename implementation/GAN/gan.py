import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.d = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.d(x)
    
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(noise_dim, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.g(z)