import torch
import matplotlib.pyplot as plt
from model.training import CustomDataset
from model.diffusionModel import DiffusionModel

dataset_name = './dataset_conceptual_captions_lite.npy'

# display samples from a dataset randomly using Gaussian distribution
def show_samples(dataset=None, num_samples=16, cols=1):
    """ Plots some samples from the dataset """
    if dataset is None:
        dataset = CustomDataset(dataset_name)

    rows = len(dataset)
    print(rows)
    plt.figure(figsize=(16,rows*2))
    for i in range(len(dataset)):
        plt.subplot(rows, cols, i+1)
        plt.axis('off')
        img, curr_label = dataset[i]
        plt.title(curr_label)
        plt.imshow(img.permute(1,2,0))
    
    plt.savefig(f'./dataset_samples.png')
    img, curr_label = dataset[int(16)]
        
show_samples()