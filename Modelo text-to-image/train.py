from model.unet import Unet
from model.training import *
from model.diffusionModel import DiffusionModel
from torch.utils.data import DataLoader
from torch.optim import Adam
from model.t5 import t5_encode_text
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np

log_file='training_progress_humanart_128.txt'
def show_msg(msg, file=log_file):
    if file is not None:
        with open(file, 'a') as f:
            f.write(msg+"\n")
    print(msg)

# training hyperparameters
batch_size = 32
n_epoch = 100
learning_rate = 1e-4
# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_feat = 512 # hidden dimension feature
max_text_len = 64 # word vector max size
height = 128 # 64x64 image
save_dir = './weights_humanart_kids_drawing_128/'
# diffusion hyperparameters
timesteps = 5000

dataset_data_path = './dataset/humanart_kids_drawing_128.npy'
# load dataset
dataset = CustomDataset(dataset_data_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

df = DiffusionModel(timesteps, height)
model = Unet(in_channels=3, n_feat=n_feat, max_text_len=max_text_len, height=height, device=device).to(device)
#model.load_state_dict(torch.load(save_dir+"model_009.pth", map_location=device))
optimizer = Adam(model.parameters(), lr=learning_rate)
open(log_file, 'w').close()
list_total_loss = []
training_steps = 0
model.train()
for epoch in range(n_epoch):
    show_msg("------------------------------------ epoch {:03d} ({} steps) ------------------------------------".format(epoch + 1, training_steps))
    total_loss = 0
    loss_list = []
    # linearly decay of learning rate
    optimizer.param_groups[0]['lr'] = learning_rate*(1-(epoch/n_epoch))
    for x_0, labels in dataloader:   # x_0: images
        optimizer.zero_grad()
        x_0 = x_0.to(device)
        # perturb data
        noise = torch.randn_like(x_0).to(device)
        t = torch.randint(1, timesteps + 1, (x_0.shape[0],)).to(device)
        x_t = df.noise_image(x_0, t, noise).to(device)
        t_emb, t_mask = t5_encode_text(labels)
        # use network to recover noise
        pred_noise = model(x_t, t/timesteps, t_emb=t_emb, t_mask=t_mask)

        # loss is measures the element-wise mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_noise, noise)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        training_steps+=1
        #if (training_steps%100) == 0:
            #print("Total train step: {}, Loss: {}".format(training_steps,loss))
            
    loss_list = np.array(loss_list)
    show_msg("Max loss: {}".format(loss_list.max()))
    show_msg("Min loss: {}".format(loss_list.min()))
    total_loss = loss_list.sum()
    show_msg("Mean loss: {}".format(loss_list.mean()))
    show_msg("Std loss: {}".format(loss_list.std()))
    show_msg("Total Loss: {}".format(total_loss))
    list_total_loss.append(total_loss)
  # save model periodically
    if epoch%10==0 or epoch == int(n_epoch-1):
        torch.save(model.state_dict(), save_dir + "model_{:03d}.pth".format(epoch+1))
        show_msg('saved model at ' + save_dir + "model_{:03d}.pth".format(epoch+1))
    
plt.figure()
plt.plot(list_total_loss)
plt.title("Total Loss vs Epoch")
plt.savefig('train.png')

show_msg("Fin del entrenamiento")