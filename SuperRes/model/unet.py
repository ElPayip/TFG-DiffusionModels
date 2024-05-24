import torch
from torch import nn
from model.layers import *


class Unet(nn.Module):
  def __init__(self, in_channels, n_feat=64, height=16):  # cfeat - context features
    super(Unet, self).__init__()
    # number of input channels, number of intermediate feature maps and number of classes
    self.h = height  #assume h == w.

    # Initialize the initial convolutional layer
    self.init_conv = Block(2*in_channels, n_feat, is_res=True) #[b, 64, 64, 64]

    # Initialize the down-sampling path of the U-Net with two levels
    self.down1 = UnetDown(n_feat, n_feat)        # down1 #[b, 64, 32, 32]
    self.down2 = UnetDown(n_feat, 2 * n_feat)    # down2 #[b, 128, 16, 16]
    self.down3 = UnetDown(2*n_feat, 4*n_feat)        # down3 #[b, 256, 8, 8]
    self.down4 = UnetDown(4*n_feat, 8* n_feat)    # down4 #[b, 512, 4, 4]
    self.down5 = UnetDown(8*n_feat, 16* n_feat)    # down4 #[b, 1024, 2, 2]

    # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.ReLU())
    self.to_vec = nn.Sequential(nn.AvgPool2d((2)), nn.ReLU())

    # Embed the timestep and context labels with a one-layer fully connected neural network
    self.timeembed1 = EmbedFC(1, 2*16*n_feat)  # (scale, shift)
    self.timeembed2 = EmbedFC(1, 2*8*n_feat)
    self.timeembed3 = EmbedFC(1, 2*4*n_feat)
    self.timeembed4 = EmbedFC(1, 2*2*n_feat)
    self.timeembed5 = EmbedFC(1, 2*n_feat)

    # Initialize the up-sampling path of the U-Net with three levels
    self.up0 = nn.Sequential(
        nn.ConvTranspose2d(16 * n_feat, 16 * n_feat, 2, 2), # up-sample
        nn.BatchNorm2d(16 * n_feat), # normalize
        nn.ReLU(),
    )                                             # up0 #[b, 1024+1024, 2, 2]
    self.up1 = UnetUp(32 * n_feat, 8 * n_feat)    # up1 #[b, 512+512, 4, 4]
    self.up2 = UnetUp(16 * n_feat, 4 * n_feat)    # up2 #[b, 256+256, 8, 8]
    self.up3 = UnetUp(8 * n_feat, 2 * n_feat)     # up3 #[b, 128+128, 16, 16]
    self.up4 = UnetUp(4 * n_feat, n_feat)         # up4 #[b, 64+64, 32, 32]
    self.up5 = UnetUp(2 * n_feat, n_feat)         # up5 #[b, 64, 64, 64]

    # Initialize the final convolutional layers to map to the same number of channels as the input image
    self.out = nn.Sequential(
        nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
        nn.ReLU(),
        nn.Conv2d(n_feat, in_channels, 3, 1, 1), # map to same number of channels as input
    )

  def forward(self, x, t, lowres):
    """
    x : (batch, n_feat, h, w) : input image
    t : (batch, n_cfeat)      : time step
    c : (batch, n_classes)    : context label
    """
    # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on
    x = torch.cat((x, lowres), dim=1)
    
    # pass the input image through the initial convolutional layer
    x = self.init_conv(x)
    # pass the result through the down-sampling path
    down1 = self.down1(x)
    down2 = self.down2(down1)
    down3 = self.down3(down2)
    down4 = self.down4(down3)
    down5 = self.down5(down4)

    # convert the feature maps to a vector and apply an activation
    hiddenvec = self.to_vec(down5)

    # embed context and timestep with (batch, 2*n_feat, 1,1) for emb1 and (batch, n_feat, 1,1) for emb2
    sc1, sh1 = self.timeembed1(t).chunk(2, dim=1) #(scale, shift)
    sc2, sh2 = self.timeembed2(t).chunk(2, dim=1)
    sc3, sh3 = self.timeembed3(t).chunk(2, dim=1)
    sc4, sh4 = self.timeembed4(t).chunk(2, dim=1)
    sc5, sh5 = self.timeembed5(t).chunk(2, dim=1)
    
    up1 = self.up0(hiddenvec)
    up2 = self.up1(sc1*up1 + sh1, down5)
    up3 = self.up2(sc2*up2 + sh2, down4)  # add and multiply embeddings
    up4 = self.up3(sc3*up3 + sh3, down3)
    up5 = self.up4(sc4*up4 + sh4, down2)  # add and multiply embeddings
    up6 = self.up5(sc5*up5 + sh5, down1)
    out = self.out(torch.cat((up6, x), 1))
    return out