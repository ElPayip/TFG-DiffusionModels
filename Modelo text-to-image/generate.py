from model.diffusionModel import DiffusionModel
from model.unet import Unet
import matplotlib.pyplot as plt
import torch


# network hyperparameters
batch_size = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_feat = 128 # hidden dimension feature
max_text_len = 128 # word vector
height = 64 # 64x64 image
save_dir = './weights/'
# diffusion hyperparameters
timesteps = 500

df = DiffusionModel(timesteps, height)
model = Unet(in_channels=3, n_feat=n_feat, max_text_len=max_text_len, height=height, device=device).to(device)
model.load_state_dict(torch.load(save_dir+"model_001.pth", map_location=device))

model.eval()
#model.train()
a = ('A fluffy white dog running across the snow',
     'One black dog with a toy and one yellow dog .',
     'A man in a yellow shirt wearing multi-colored plastic jewelry .',
     'Seven people are jumping on the air , along the shore .')
samples = df.sample_ddpm_context(model, 4, a)
df.draw_samples_process(path='./generated_images/')
"""
samples = samples.detach().cpu().numpy()
for i, img in enumerate(samples):
    plt.subplot(1, 4, i + 1)
    plt.axis('off')
    img = df.unorm(img)
    plt.imshow(img.permute(1,2,0))
"""

print('Imagenes generadas')