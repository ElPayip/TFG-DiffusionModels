from torch import nn
from model.layers import *

class Unet(nn.Module):
  def __init__(self, in_channels, n_feat=64, max_text_len=10, height=28, width=56, cond_drop_prob=0.1, device=None):  # cfeat - context features
    super(Unet, self).__init__()
    # number of input channels, number of intermediate feature maps and number of classes
    self.in_channels = in_channels
    self.n_feat = n_feat
    self.h = height  #assume h == w.
    self.cond_drop_prob = cond_drop_prob
    self.device = device
    time_cond_dim = 4*n_feat

    # Embed the timestep and context labels with a one-layer fully connected neural network
    self.time_cond = TimeConditioningLayer(n_feat, time_cond_dim)
    self.text_cond = TextConditioningLayer(n_feat, max_text_len=max_text_len)

    # Initialize the initial convolutional layer                                                                # in    (b, 3, 28, 56)
    self.init_conv = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.Conv2d(in_channels, n_feat//8, kernel_size=3, padding=1),                # init  (b, 32, 28, 56)
                                   nn.ReLU(),
                                   nn.Conv2d(n_feat//8, n_feat//4, kernel_size=7, padding=3),
                                   nn.ReLU(),
                                   nn.Conv2d(n_feat//4, n_feat//2, kernel_size=15, padding=7),
                                   )

    # Initialize the down-sampling path of the U-Net
    self.down = nn.ModuleList([
                 UnetDown(n_feat//2, n_feat, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, device=device),                   # down2 (b, 64, 14, 28)
                 UnetDown(n_feat, n_feat*2, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, self_attn=True, device=device),
                 UnetDown(n_feat*2, n_feat*4, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, self_attn=True, device=device)])                   # down3 (b, 128, 7, 28)

    self.mid1 = Block(n_feat*4, n_feat*4, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, device=device)
    self.mid_attn1 = TransformerBlock(n_feat*4, context_dim=n_feat)
    self.mid2 = Block(n_feat*4, n_feat*4, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, device=device)
    self.mid_attn2 = TransformerBlock(n_feat*4, context_dim=n_feat)
    self.mid3 = Block(n_feat*4, n_feat*4, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, device=device)         # mid  (b, 128, 7, 28)

    # Initialize the up-sampling path of the U-Net
    self.up = nn.ModuleList([UnetUp(8*n_feat, 2*n_feat, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, self_attn=True, device=device),       # up1 (b, 128, 7, 28)
               UnetUp(n_feat*4, n_feat, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, self_attn=True, device=device),
               UnetUp(n_feat*2, n_feat//2, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, device=device),# up2 (b, 128, 14, 16)
               ])

    # Initialize the final convolutional layers to map to the same number of channels as the input image
    self.out = nn.Sequential(
        nn.BatchNorm2d(n_feat//2),
        nn.Conv2d(n_feat//2, n_feat//4, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(n_feat//4, self.in_channels, 3, 1, 1), # map to same number of channels as input
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
    x = x_down[-1]

    x = self.mid1(x, t, c)
    x = self.mid_attn1(x, c)
    x = self.mid2(x, t, c)
    x = self.mid_attn2(x, c)
    x = self.mid3(x, t, c)

    for i, up in enumerate(self.up):
      x = up(x, x_down[-i-1], t, c)

    return self.out(x)