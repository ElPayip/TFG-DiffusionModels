from torch import nn
from model.layers import *


class Unet(nn.Module):
  def __init__(self, in_channels, n_feat=512, max_text_len=512, height=64, cond_drop_prob=0.1, device=None,
               x_attns=(True, True, True), self_attns=(False, True, True)):  # cfeat - context features
    super(Unet, self).__init__()
    # number of input channels, number of intermediate feature maps and number of classes
    self.in_channels = in_channels
    self.n_feat = n_feat
    self.h = height  #assume h == w.
    self.cond_drop_prob = cond_drop_prob
    self.device = device
    time_cond_dim = 4*n_feat
    assert len(x_attns) == len(self_attns), "'x_attns' y 'self_attns' deben ser del mismo tamaño"
    n_floors = len(x_attns)

    # Embed the timestep and context labels with a one-layer fully connected neural network
    self.time_cond = TimeConditioningLayer(n_feat, time_cond_dim)
    self.text_cond = TextConditioningLayer(n_feat, max_text_len=max_text_len)

    # Initialize the initial convolutional layer                                                                # in    (b, 3, 64, 64)
    self.init_conv = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.Conv2d(in_channels, n_feat//(2 * 2**n_floors), kernel_size=3, padding=1),                # init  (b, 32, 64, 64)
                                   nn.ReLU(),
                                   nn.Conv2d(n_feat//(2 * 2**n_floors), n_feat//(1.5 * 2**n_floors), kernel_size=7, padding=3),
                                   nn.ReLU(),
                                   nn.Conv2d(n_feat//(1.5 * 2**n_floors), n_feat//(2**n_floors), kernel_size=15, padding=7),
                                   )

    # Initialize the down-sampling path of the U-Net
    self.down = nn.ModuleList()
    for i in range(n_floors):
      self.down.append(UnetDown(n_feat//(2**(n_floors-i)), n_feat//(2**(n_floors-i-1)), 
                                time_cond_dim=time_cond_dim, text_cond_dim=n_feat if x_attns[i] else None, 
                                device=device, self_attn=self_attns[i]))

    self.mid1 = Block(n_feat, n_feat, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, device=device)
    self.mid_attn1 = TransformerBlock(n_feat, context_dim=n_feat)
    self.mid2 = Block(n_feat, n_feat, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, device=device)
    self.mid_attn2 = TransformerBlock(n_feat, context_dim=n_feat)
    self.mid3 = Block(n_feat, n_feat, time_cond_dim=time_cond_dim, text_cond_dim=n_feat, device=device)         # mid  (b, 512, 4, 4)

    # Initialize the up-sampling path of the U-Net
    self.up = nn.ModuleList()
    for i in range(n_floors-1, -1, -1):
      self.up.append(UnetUp(2 * n_feat//(2**(n_floors-i-1)), n_feat//(2**(n_floors-i)), 
                                time_cond_dim=time_cond_dim, text_cond_dim=n_feat if x_attns[i] else None, 
                                device=device, self_attn=self_attns[i]))

    # Initialize the final convolutional layers to map to the same number of channels as the input image
    self.out = nn.Sequential(
        nn.BatchNorm2d(n_feat//(2**n_floors)),
        nn.Conv2d(n_feat//(2**n_floors), n_feat//(1.5 * 2**n_floors), 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(n_feat//(1.5 * 2**n_floors), n_feat//(2 * 2**n_floors), 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(n_feat//(2 * 2**n_floors), self.in_channels, 3, 1, 1), # map to same number of channels as input
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