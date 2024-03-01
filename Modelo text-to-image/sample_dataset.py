import torch
import matplotlib.pyplot as plt
from model.training import CustomDataset
from model.diffusionModel import DiffusionModel

dataset_name = 'humanart_kids_drawing_256'

# display samples from a dataset randomly using Gaussian distribution
def show_samples(dataset=None, num_samples=400, cols=1):
    """ Plots some samples from the dataset """
    if dataset is None:
        dataset = CustomDataset('./dataset/'+dataset_name+'.npy')
    df = DiffusionModel(500, 64)

    rows = int(num_samples / cols)
    if num_samples%cols!=0:
        rows += 1
    plt.figure(figsize=(16,rows*2))
    random_idx = torch.randint(0,len(dataset),(num_samples,))
    for i, idx in enumerate(random_idx):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        img, curr_label = dataset[int(idx)]
        img = df.unorm(img)
        plt.title(curr_label)
        plt.imshow(img.permute(1,2,0))
    
    plt.savefig(f'./dataset_samples/dataset_samples_'+dataset_name+'.png')
    img, curr_label = dataset[int(16)]
        
show_samples(cols=2)