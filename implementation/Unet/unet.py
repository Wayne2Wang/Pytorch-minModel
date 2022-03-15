import torch
import torch.nn as nn

class Unet(nn.Module):

    def __init__(self, in_channel, label_dim):
        super().__init__()
        # Contracting path (input image 2048x1024)
        self.two_conv1 = TwoConvLayer(in_channel, 64, 64) # 2044x1020
        self.maxpool1 = nn.MaxPool2d(2,2) # 1022x510
        self.two_conv2 = TwoConvLayer(64, 128, 128) # 1018x506
        self.maxpool2 = nn.MaxPool2d(2,2) # 509x253
        self.two_conv3 = TwoConvLayer(128, 256, 256) # 505x249
        self.maxpool3 = nn.MaxPool2d(2,2) # 252x124
        self.two_conv4 = TwoConvLayer(256, 512, 512) # 248x120
        self.maxpool4 = nn.MaxPool2d(2,2)  # 124x60
        self.two_conv5 = TwoConvLayer(512, 1024, 1024) # 120x56

        # Expansive path
        self.upconv1 = UpConv(1024, 512, 512) # 236x108 
        self.upconv2 = UpConv(512, 256, 256) # 468x212
        self.upconv3 = UpConv(256, 128, 128) # 932x420
        self.upconv4 = UpConv(128, 64, 64) # 1860x836

        # 1x1 conv
        self.final_conv = nn.Conv2d(64, label_dim, kernel_size=1, padding=0)

    def forward(self, x):
        # Contracting path
        prev1 = self.two_conv1(x)
        prev2 = self.two_conv2(self.maxpool1(prev1))
        prev3 = self.two_conv3(self.maxpool2(prev2))
        prev4 = self.two_conv4(self.maxpool3(prev3))
        x = self.two_conv5(self.maxpool4(prev4))

        # Expansive path
        x = self.upconv1(x, prev4)
        x = self.upconv2(x, prev3)
        x = self.upconv3(x, prev2)
        x = self.upconv4(x, prev1)

        # Final output
        logits = self.final_conv(x)
        return logits


# The repeated application of two 3x3 convolution and ReLU
class TwoConvLayer(nn.Module):

    def __init__(self, in_channel, mid_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=0),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layer(x)

# Upsampling, concatenate with cropped previous feature, two convolution
class UpConv(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channel,in_channel//2, kernel_size=2, stride=2)
        self.two_conv = TwoConvLayer(in_channel, mid_channel, out_channel)

    def forward(self, x , prev):
        up_x = self.up_sample(x)

        # crop previous feature to have the same spatial dim as x
        _,_,hx,wx = up_x.shape
        _,_,hp,wp = prev.shape
        cropped_prev = prev[:,:,hp//2-hx//2:hp//2-hx//2+hx, wp//2-wx//2:wp//2-wx//2+wx]

        combine = torch.concat((up_x, cropped_prev), dim=1)
        output = self.two_conv(combine)
        return output