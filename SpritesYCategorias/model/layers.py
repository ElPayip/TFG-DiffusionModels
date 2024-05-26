import torch
from torch import nn


class Block(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
    super().__init__()

    # Check if input and output channels are not the same for the residual connection
    self.non_same_channels = in_channels != out_channels

    # Boolean to control if use or not residual connection
    self.is_res = is_res

    # First convolutional layer
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
         nn.BatchNorm2d(out_channels),   # Batch normalization
        nn.ReLU(),   # GELU activation function
        )

    # Second convolutional layer
    self.conv2 = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
         nn.BatchNorm2d(out_channels),   # Batch normalization
        nn.ReLU(),   # GELU activation function
        )

  def forward(self, x):
    # If using residual connection
    if self.is_res:
      x1 = self.conv1(x)
      x2 = self.conv2(x1)

      # If input and output channels are not the same, apply a 1x1 convolutional layer to match dimensions before adding residual connection
      if self.non_same_channels:
        shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
        x = shortcut(x) + x2
      out = x + x2
    # If not using residual connection, return output of second convolutional layer
    else:
      x1 = self.conv1(x)
      out = self.conv2(x1)
    return out

class UnetUp(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UnetUp, self).__init__()
    # The model consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
    self.model = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
        Block(out_channels, out_channels),
        Block(out_channels, out_channels)
        )

  def forward(self, x, skip):
    # Concatenate the input tensor x with the skip connection tensor along the channel dimension
    x = torch.cat((x, skip), 1)

    # Pass the concatenated tensor through the sequential model and return the output
    x = self.model(x)
    return x


class UnetDown(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UnetDown, self).__init__()
    # Each block consists of two Block layers, followed by a MaxPool2d layer for downsampling
    self.model = nn.Sequential(
        Block(in_channels, out_channels),
        Block(out_channels, out_channels),
        nn.MaxPool2d(2)
        )

  def forward(self, x):
    return self.model(x)

class EmbedFC(nn.Module):
  def __init__(self, input_dim, emb_dim):
    super(EmbedFC, self).__init__()
    #This class defines a generic one layer feed-forward neural network for embedding input data of
    #dimensionality input_dim to an embedding space of dimensionality emb_dim.
    self.input_dim = input_dim
    self.emb_dim = emb_dim
    self.model = nn.Sequential(
        nn.Linear(input_dim, emb_dim),
        nn.ReLU(),
        nn.Linear(emb_dim, emb_dim)
        )

  def forward(self, x):
    # flatten the input tensor
    x = x.view(-1, self.input_dim)
    # apply the model layers to the flattened tensor
    return self.model(x).view(-1, self.emb_dim, 1, 1)