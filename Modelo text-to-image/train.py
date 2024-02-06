from model.unet import Unet
from model.t5 import t5_encode_text
from model.training import *
from model.diffusionModel import DiffusionModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import torch

# training hyperparameters
batch_size = 32
n_epoch = 50
learning_rate = 1e-2
# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_feat = 128 # hidden dimension feature
max_text_len = 512 # word vector
height = 64 # 64x64 image
save_dir = './weights/'
# diffusion hyperparameters
timesteps = 500


dataset_data_path = './datasets/dataset_conceptual_captions_lite.npy'
# load dataset
dataset = CustomDataset(dataset_data_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

df = DiffusionModel(timesteps, height)
model = Unet(in_channels=3, n_feat=n_feat, max_text_len=max_text_len, height=height, device=device).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=0.0105,
                                                steps_per_epoch=len(dataloader),
                                                epochs=n_epoch,
                                                pct_start=0.43,
                                                div_factor=10,
                                                final_div_factor=1000,
                                                three_phase=True, verbose=False)


training_steps = 0
model.train()
for epoch in range(n_epoch):
  print("------ epoch {:03d} ------".format(epoch + 1))

  total_loss = 0
  for x_0, labels in dataloader:   # x_0: images

    optimizer.zero_grad()
    x_0 = x_0.to(device)

    # perturb data
    noise = torch.randn_like(x_0).to(device)
    t = torch.randint(1, timesteps + 1, (x_0.shape[0],)).to(device)
    x_t = df.noise_image(x_0, t, noise).to(device)
    t_emb, t_mask = t5_encode_text(labels)

    # use network to recover noise
    pred_noise = model(x_t, t / timesteps, t_emb=t_emb, t_mask=t_mask)

    # loss is measures the element-wise mean squared error between the predicted and true noise
    loss = F.mse_loss(pred_noise, noise)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    scheduler.step()
    training_steps+=1
    if (training_steps%100) == 0:
      print("Total train step: {}, Loss: {}".format(training_steps,loss))

  print("Total Loss: {}".format(total_loss))
  # save model periodically
  if epoch%5==0 or epoch == int(n_epoch-1):
    torch.save(model.state_dict(), save_dir + "model_{:03d}.pth".format(epoch + 1))
    print('saved model at ' + save_dir + "model_{:03d}.pth".format(epoch + 1))

print("Fin del entrenamiento")