from model.unet import Unet
from model.t5 import t5_encode_text
from model.training import *
from model.diffusionModel import DiffusionModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

# training hyperparameters
batch_size = 128
n_epoch = 100
learning_rate = 1e-3
# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
n_feat = 512 # hidden dimension feature
max_text_len = 128 # word vector max size
height = 64 # 64x64 image
save_dir = './weights/'
# diffusion hyperparameters
timesteps = 500

dataset_data_path = './dataset/Flickr30k_dataset.npy'
# load dataset
dataset = CustomDataset(dataset_data_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

df = DiffusionModel(timesteps, height)
model = Unet(in_channels=3, n_feat=n_feat, max_text_len=max_text_len, height=height, device=device).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)

list_total_loss = []
training_steps = 0
for epoch in range(n_epoch):
    print("------------------------------------ epoch {:03d} ------------------------------------".format(epoch + 1))
    model.train()

    total_loss = 0
    max_loss=0
    min_loss=100
    # linearly decay of learning rate
    optimizer.param_groups[0]['lr'] = learning_rate*(1-(epoch/n_epoch))
    print(f'lr: {optimizer.param_groups[0]["lr"]}')
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
        max_loss = max(loss.item(), max_loss)
        min_loss = min(loss.item(), min_loss)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        training_steps+=1
        if (training_steps%100) == 0:
            print("Total train step: {}, Loss: {}".format(training_steps,loss))
    print("Min loss: {}, Max loss: {}".format(min_loss,max_loss))
    print("Total Loss: {}".format(total_loss))
    list_total_loss.append(total_loss)
  # save model periodically
    if epoch%5==0 or epoch == int(n_epoch-1):
        torch.save(model.state_dict(), save_dir + "model_{:03d}.pth".format(epoch + 1))
        print('saved model at ' + save_dir + "model_{:03d}.pth".format(epoch + 1))
    
plt.figure()
plt.plot(list_total_loss)
plt.title("Total Loss vs Epoch")
plt.savefig('train.png')

print("Fin del entrenamiento")