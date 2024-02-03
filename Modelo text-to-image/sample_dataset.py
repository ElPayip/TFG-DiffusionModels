import torch
import matplotlib.pyplot as plt
from model.training import CustomDataset

# display samples from a dataset randomly using Gaussian distribution
def show_samples(dataset=None, num_samples=40, cols=1):
    """ Plots some samples from the dataset """
    if dataset is None:
        dataset = CustomDataset('/dataset/dataset_conceptual_captions.npy')

    rows = int(num_samples / cols)
    if num_samples%cols!=0:
        rows += 1
    plt.figure(figsize=(16,rows*2))
    random_idx = torch.randint(0,len(dataset),(num_samples,))
    for i, idx in enumerate(random_idx):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        img, curr_label = dataset[int(idx)]
        #img = df.unorm(img)
        plt.title(curr_label)
        plt.imshow(img.permute(1,2,0))

print('Use show_samples(dataset=None, num_samples=40, cols=1) to plot some samples from the dataset')