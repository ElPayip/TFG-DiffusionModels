import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SpritesDataset(Dataset):
  def __init__(self, sfilename, lfilename, transform=None, null_context=False):
    self.sprites = np.load(sfilename)
    self.slabels = np.load(lfilename)
    self.null_context = null_context
    self.sprites_shape = self.sprites.shape
    self.slabel_shape = self.slabels.shape
    if transform is None:
      self.transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(0.5,0.5)
          ])
    else:
      self.transform = transform

  # Return the number of images in the dataset
  def __len__(self):
    return len(self.sprites)

  # Get the image and label at a given index
  def __getitem__(self, idx):
    # Return the image and label as a tuple
    image = self.transform(self.sprites[idx])
    if self.null_context:
      label = torch.tensor(0).to(torch.float32)
    else:
      label = torch.tensor(self.slabels[idx], dtype=torch.float32)
    return (image, label)

  # Get the shapes of datas and labels
  def getshapes(self):
    # return shapes of data and labels
    return self.sprites_shape, self.slabel_shape