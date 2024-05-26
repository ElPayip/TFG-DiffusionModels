import torch
from torch import nn
from model.layers import *


class Unet(nn.Module):
  def __init__(self, in_channels, n_feat=64, n_cfeat=10, height=16):  # cfeat - context features
    super(Unet, self).__init__()
    # number of input channels, number of intermediate feature maps and number of classes
    self.in_channels = in_channels
    self.n_feat = n_feat
    self.n_cfeat = n_cfeat
    self.h = height  #assume h == w.

    # Initialize the initial convolutional layer
    self.init_conv = Block(in_channels, n_feat, is_res=True)

    # Initialize the down-sampling path of the U-Net with two levels
    self.down1 = UnetDown(n_feat, n_feat)        # down1 #[10, 64, 8, 8]
    self.down2 = UnetDown(n_feat, 2 * n_feat)    # down2 #[10, 128, 4, 4]

    # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.ReLU())
    self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.ReLU())

    # Embed the timestep and context labels with a one-layer fully connected neural network
    self.timeembed1 = EmbedFC(1, 2*n_feat)
    self.timeembed2 = EmbedFC(1, 1*n_feat)
    self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
    self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

    # Initialize the up-sampling path of the U-Net with three levels
    self.up0 = nn.Sequential(
        nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # up-sample
        nn.BatchNorm2d(2 * n_feat), # normalize
        nn.ReLU(),
    )                                             # up0 #[10, 128+128, 4, 4]
    self.up1 = UnetUp(4 * n_feat, n_feat)         # up1 #[10, 64+64, 8, 8]
    self.up2 = UnetUp(2 * n_feat, n_feat)         # up2 #[10, 64, 16, 16]

    # Initialize the final convolutional layers to map to the same number of channels as the input image
    self.out = nn.Sequential(
        nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
        nn.ReLU(),
        nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map to same number of channels as input
    )

  def forward(self, x, t, c=None):
    """
    x : (batch, n_feat, h, w) : input image
    t : (batch, n_cfeat)      : time step
    c : (batch, n_classes)    : context label
    """
    # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on
    # pass the input image through the initial convolutional layer
    x = self.init_conv(x)
    # pass the result through the down-sampling path
    down1 = self.down1(x)
    down2 = self.down2(down1)

    # convert the feature maps to a vector and apply an activation
    hiddenvec = self.to_vec(down2)
    # mask out context if context_mask == 1
    if c is None:
      c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

    # embed context and timestep with (batch, 2*n_feat, 1,1) for emb1 and (batch, n_feat, 1,1) for emb2
    cemb1 = self.contextembed1(c)
    temb1 = self.timeembed1(t)
    cemb2 = self.contextembed2(c)
    temb2 = self.timeembed2(t)

    up1 = self.up0(hiddenvec)
    up2 = self.up1(cemb1*up1 + temb1, down2)  # add and multiply embeddings
    up3 = self.up2(cemb2*up2 + temb2, down1)
    out = self.out(torch.cat((up3, x), 1))
    return out