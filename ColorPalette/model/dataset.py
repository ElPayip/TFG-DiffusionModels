import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, filename, transform=None):
        file = np.load(filename, allow_pickle = True)
        self.dataset = file.item()['dataset']
        self.len = file.item()['length']
        self.name = file.item()['name']
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.5140147805213928,0.3199092149734497)
            ])
        else:
            self.transform = transform

    # Return the number of images in the dataset
    def __len__(self):
        return self.len

    # Get the image and label at a given index

    def __getitem__(self, idx):
        # Return the image and label as a tuple
        image = self.transform(self.dataset[idx]['img'])
        return image

    # Get the shapes of datas and labels
    def getshapes(self):
        # return shapes of data and labels
        return self.dataset['shape']