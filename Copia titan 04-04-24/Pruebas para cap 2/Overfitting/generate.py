from model.diffusionModel import DiffusionModel
from model.unet import Unet
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch


# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_feat = 512 # hidden dimension feature
max_text_len = 128 # word vector
height = 64 # 64x64 image
save_dir = './weights/'
# diffusion hyperparameters
timesteps = 5000

df = DiffusionModel(timesteps, height)
model = Unet(in_channels=3, n_feat=n_feat, max_text_len=max_text_len, height=height, device=device).to(device)
model.load_state_dict(torch.load(save_dir+"model_9901.pth", map_location=device))

model.eval()
#model.train()
"""
a = ('beautiful dog with glasses portrait - isolated over a white background',
     'swallow bird isolated on a white background',
     'blue alarm clock on the wooden surface against the orange background photo',
     'beautiful pink lotus on a black background')"""
a = ('dog with glasses portrait',
     'dog with glasses portrait',
     'dog with glasses portrait',
     'dog with glasses portrait')

samples = df.sample_ddpm_context(model, len(a), a,save_rate=timesteps//20)
#df.draw_samples_process(path='./generated_images/')
df.draw_samples(prompts=a, path='./generated_images/')
print(samples[0])

print('Imagenes generadas')