import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_c, batch_norm=False):
        super().__init__()
        if batch_norm:
            self.d = nn.Sequential(
                # input image shape: 64
                nn.Conv3d(img_c,64, kernel_size=4, stride=4), # 16
                nn.BatchNorm3d(64),
                nn.LeakyReLU(0.2),
                nn.Conv3d(64,128, kernel_size=4, stride=2, padding=1), # 8
                nn.BatchNorm3d(128),
                nn.LeakyReLU(0.2),
                nn.Conv3d(128,256, kernel_size=4, stride=2, padding=1), # 4
                nn.BatchNorm3d(256),
                nn.LeakyReLU(0.2),
                nn.Conv3d(256,1, kernel_size=4, stride=1), # 1
            )
        else:
            self.d = nn.Sequential(
                # input image shape: 64
                nn.Conv3d(img_c,64, kernel_size=4, stride=4), # 16
                nn.LeakyReLU(0.2),
                nn.Conv3d(64,128, kernel_size=4, stride=2, padding=1), # 8
                nn.LeakyReLU(0.2),
                nn.Conv3d(128,256, kernel_size=4, stride=2, padding=1), # 4
                nn.LeakyReLU(0.2),
                nn.Conv3d(256,1, kernel_size=4, stride=1), # 1
            )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # using BCE with logits, skip sigmoid here
        pre_act = self.d(x)
        out = pre_act
        return pre_act, out


class Generator(nn.Module):
    def __init__(self, noise_dim, img_c, batch_norm=False):
        super().__init__()
        self.batch_norm=batch_norm
        self.fc1 = nn.Linear(noise_dim, 256*8*8*8) # 8
        self.trans_conv1 = nn.ConvTranspose3d(256, 64, kernel_size=4, stride=2, padding=1) # 16
        self.trans_conv2 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1) # 32
        self.trans_conv3 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1) # 64
        self.conv1 = nn.Conv3d(16,img_c, kernel_size=1, stride=1) # 64

        if batch_norm:
            self.bn1 = nn.BatchNorm3d(64)
            self.bn2 = nn.BatchNorm3d(32)
            self.bn3 = nn.BatchNorm3d(16)

        # activations
        self.relu = nn.ReLU()
        self.tanh  = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        if self.batch_norm:
            z = self.fc1(z)
            z = z.reshape(-1, 256,8,8,8)
            z = self.relu(self.bn1(self.trans_conv1(z)))
            z = self.relu(self.bn2(self.trans_conv2(z)))
            z = self.relu(self.bn3(self.trans_conv3(z)))
            z = self.relu(self.conv1(z))
        else:
            z = self.fc1(z)
            z = z.reshape(-1, 256,8,8,8)
            z = self.relu(self.trans_conv1(z))
            z = self.relu(self.trans_conv2(z))
            z = self.relu(self.trans_conv3(z))
            z = self.tanh(self.conv1(z))
        return z



def main():
    spatial_dim = 64
    noise_dim = 200
    img_c = 1
    batch_size = 2
    dummy_img_batch = torch.randn(batch_size,img_c,spatial_dim,spatial_dim,spatial_dim)
    dummy_noise = torch.randn(batch_size, noise_dim)
    
    gen = Generator(noise_dim, img_c, batch_norm=True)
    disc = Discriminator(img_c, batch_norm=True)
    fake = gen(dummy_noise)
    score = disc(fake)
    print('Noise batch shape {}'.format(dummy_noise.shape))
    print('Image batch shape {}'.format(dummy_img_batch.shape))
    print('Generator shape: {}'.format(fake.shape))
    print('Discriminator shape: {}'.format(score.shape))

if __name__ == '__main__':
    main()