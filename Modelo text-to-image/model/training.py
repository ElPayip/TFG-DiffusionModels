from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class CustomDataset(Dataset):
  def __init__(self, filename, transform=None):
    file = np.load(filename, allow_pickle = True)
    self.dataset = file.item()['dataset']
    self.len = file.item()['length']

    if transform is None:
      self.transform = transforms.ToTensor()
    else:
      self.transform = transform

  # Return the number of images in the dataset
  def __len__(self):
    return self.len

  # Get the image and label at a given index
  def __getitem__(self, idx):
    # Return the image and label as a tuple
    image = self.transform(self.dataset[idx]['img'])
    return (image, self.dataset[idx]['label'])

  # Get the shapes of datas and labels
  def getshapes(self):
    # return shapes of data and labels
    return self.dataset['shape']