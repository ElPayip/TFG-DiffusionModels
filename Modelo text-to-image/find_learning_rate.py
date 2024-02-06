from model.unet import Unet
from model.training import *
from model.diffusionModel import DiffusionModel
from model.training import find_lr
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch

# training hyperparameters
batch_size = 64
learning_rate = 1e-3
# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_feat = 128 # hidden dimension feature
max_text_len = 512 # word vector
height = 64 # 64x64 image
# diffusion hyperparameters
timesteps = 500


dataset_data_path = './datasets/Flickr8k_dataset.npy'
# load dataset
dataset = CustomDataset(dataset_data_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

df = DiffusionModel(timesteps, height)
model = Unet(in_channels=3, n_feat=n_feat, max_text_len=max_text_len, height=height, device=device).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)


find_lr(model, optimizer, dataloader, timesteps, df, start_val=1e-6, end_val=1, device=device)