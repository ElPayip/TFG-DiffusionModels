from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
from model.t5 import t5_encode_text


class CustomDataset(Dataset):
  def __init__(self, filename, transform=None):
    file = np.load(filename, allow_pickle = True)
    self.dataset = file.item()['dataset']
    self.len = file.item()['length']
    self.name = file.item()['name']
    if self.name == 'flickr30k':
        self.label_per_img = len(self.dataset[0]['label'])
    if transform is None:
      self.transform = transforms.Compose([
          transforms.ToTensor(),
          ])
    else:
      self.transform = transform

  # Return the number of images in the dataset
  def __len__(self):
    return self.len

  # Get the image and label at a given index
  def __getitem__(self, idx):
    # Return the image and label as a tuple
    """if self.name == 'flickr30k':
        label_idx = idx % self.label_per_img
        img_idx = idx // self.label_per_img
        image = self.transform(self.dataset[img_idx]['img'])
        caption = self.dataset[img_idx]['label'][label_idx]
    else:"""
    image = self.transform(self.dataset[idx]['img'])
    caption = self.dataset[idx]['label']
    return (image, caption)

  # Get the shapes of datas and labels
  def getshapes(self):
    # return shapes of data and labels
    return self.dataset['shape']