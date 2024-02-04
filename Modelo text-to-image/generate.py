from model.diffusionModel import DiffusionModel
from model.unet import Unet
import matplotlib.pyplot as plt
import torch


# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_feat = 128 # hidden dimension feature
max_text_len = 64 # word vector
height = 64 # 64x64 image
save_dir = './weights/flick8k/'
# diffusion hyperparameters
timesteps = 500

df = DiffusionModel(timesteps, height)
model = Unet(in_channels=3, n_feat=n_feat, max_text_len=max_text_len, height=height).to(device)
model.load_state_dict(torch.load(save_dir+"model_163.pth", map_location=device))

model.eval()
a = ['A brown dog playing in mud .',
     'A kids .',
     'A dog.',
     'A little girl in a pink dress going into a wooden cabin .']
samples = df.sample_ddpm_context(model, 4, a)
samples = samples.detach().cpu().numpy()
for i, img in enumerate(samples):
    plt.subplot(1, 4, i + 1)
    plt.axis('off')
    img = df.unorm(img)
    plt.imshow(img.permute(1,2,0))