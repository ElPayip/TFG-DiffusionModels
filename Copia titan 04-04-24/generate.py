from model.diffusionModel import DiffusionModel
from model.unet import Unet
import matplotlib.pyplot as plt
import torch


# network hyperparameters
batch_size = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_feat = 512 # hidden dimension feature
max_text_len = 64 # word vector
height = 64 # 64x64 image
save_dir = './weights_humanart_digital_art/'
# diffusion hyperparameters
timesteps = 1000

df = DiffusionModel(timesteps, height)
model = Unet(in_channels=3, n_feat=n_feat, max_text_len=max_text_len, height=height, device=device).to(device)
model.load_state_dict(torch.load(save_dir+"model_501.pth", map_location=device))

model.eval()
#model.train()

a = ('digital_art, a man standing on a hill looking at a landscape',
     'digital_art, a girl standing in front of a dragon',
     'digital_art, the forest wallpaper')

samples = df.sample_ddpm_context(model, 3, a,save_rate=timesteps//20)
df.draw_samples_process(path='./generated_images/')
print(samples[0])
"""
samples = samples.detach().cpu().numpy()
for i, img in enumerate(samples):
    plt.subplot(1, 4, i + 1)
    plt.axis('off')
    img = df.unorm(img)
    plt.imshow(img.permute(1,2,0))
"""

print('Imagenes generadas')