from torch import nn
from layers import *


class Unet(nn.Module):
  def __init__(self, in_channels, n_feat=128, n_cfeat=512, height=64, cond_drop_prob=0.1, device=None):  # cfeat - context features
    super(Unet, self).__init__()
    # number of input channels, number of intermediate feature maps and number of classes
    self.in_channels = in_channels
    self.n_feat = n_feat
    self.n_cfeat = n_cfeat
    self.h = height  #assume h == w.
    self.cond_drop_prob = cond_drop_prob
    self.device = device
    time_cond_dim = 4*n_feat

    # Embed the timestep and context labels with a one-layer fully connected neural network
    self.time_cond = TimeConditioningLayer(n_feat, time_cond_dim)
    self.text_cond = TextConditioningLayer(n_feat)

    # Initialize the initial convolutional layer                                                                # in    (b, 3, 64, 64)
    self.init_conv = Block(in_channels, n_feat//4)                                                              # init  (b, 32, 64, 64)

    # Initialize the down-sampling path of the U-Net
    self.down = nn.ModuleList([UnetDown(n_feat//4, n_feat//2, time_cond_dim=time_cond_dim),                     # down1 (b, 64, 32, 32)
                 UnetDown(n_feat//2, n_feat, time_cond_dim=time_cond_dim),                                      # down2 (b, 128, 16, 16)
                 UnetDown(n_feat, n_feat, time_cond_dim=time_cond_dim, text_cond_dim=n_feat)])                  # down3 (b, 128, 8, 8)

    self.mid = nn.ModuleList([Block(n_feat, n_feat, time_cond_dim=time_cond_dim, text_cond_dim=n_feat),
                             Block(n_feat, n_feat, time_cond_dim=time_cond_dim, text_cond_dim=n_feat)])         # mid  (b, 128, 8, 8)

    # Initialize the up-sampling path of the U-Net
    self.up = nn.ModuleList([UnetUp(2*n_feat, n_feat, time_cond_dim=time_cond_dim, text_cond_dim=n_feat),       # up1 (b, 128, 16, 16)
               UnetUp(2*n_feat, n_feat//2, time_cond_dim=time_cond_dim),                                        # up2 (b, 64, 32, 32)
               UnetUp(n_feat, n_feat//4, time_cond_dim=time_cond_dim)])                                         # up3 (b, 32, 64, 64)

    # Initialize the final convolutional layers to map to the same number of channels as the input image
    self.out = nn.Sequential(
        nn.Conv2d(n_feat//4, n_feat//8, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(n_feat//8, self.in_channels, 3, 1, 1), # map to same number of channels as input
    )                                                                                                    # out (b, 3, 64, 64)


  def forward(self, x, time, t_emb=None, t_mask=None):
    """
    x : (batch, 3, h, w) : input image
    t : (batch,)      : time step
    t_emb  : (batch, n_words, n_cfeat)   : text embed
    t_mask : (batch, n_words, n_cfeat)   : text mask
    """
    # x is the input image, t_emb is the context label, t is the timestep, context_mask says which samples to block the context on
    t, time_tokens = self.time_cond(time)
    t, c = self.text_cond(text_embeds=t_emb, text_mask=t_mask, cond_drop_prob=self.cond_drop_prob,
                     device=self.device, t=t, time_tokens=time_tokens)

    # pass the input image through the initial convolutional layer
    x = self.init_conv(x)
    # pass the result through the down-sampling path
    x_down = [x]
    for i, down in enumerate(self.down):
      x_down.append(down(x_down[i], t, c))
    x_down.pop(0)

    x = self.mid[0](x_down[-1], t, c)
    x = self.mid[1](x, t, c)

    for i, up in enumerate(self.up):
      x = up(x, x_down[-i-1], t, c)

    return self.out(x)