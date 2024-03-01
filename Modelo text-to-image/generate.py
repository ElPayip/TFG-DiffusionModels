from model.diffusionModel import DiffusionModel
from model.unet import Unet
import matplotlib.pyplot as plt
import torch


# network hyperparameters
batch_size = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_feat = 512 # hidden dimension feature
max_text_len = 64 # word vector
height = 128 # 64x64 image
save_dir = './weights_humanart_kids_drawing_128/'
# diffusion hyperparameters
timesteps = 5000

df = DiffusionModel(timesteps, height)
model = Unet(in_channels=3, n_feat=n_feat, max_text_len=max_text_len, height=height, device=device).to(device)
model.load_state_dict(torch.load(save_dir+"model_100.pth", map_location=device))

model.eval()
#model.train()

a = ('kids_drawing, a cartoon girl singing into a microphone',
     'kids_drawing, a cartoon girl singing into a microphone',
     'kids_drawing, a cartoon girl singing into a microphone',
     'kids_drawing, a cartoon girl singing into a microphone')

samples = df.sample_ddpm_context(model, 4, a,save_rate=800)
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