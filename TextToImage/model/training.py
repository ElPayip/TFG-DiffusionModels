from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
from model.t5 import t5_encode_text


class CustomDataset(Dataset):
  def __init__(self, filename, transform=None):
    file = np.load(filename, allow_pickle = True)
    self.dataset = file.item()['dataset']
    self.len = file.item()['length']
    self.name = file.item()['name']
    if self.name == 'flickr30k':
        self.label_per_img = len(self.dataset[0]['label'])
    if transform is None:
      self.transform = transforms.Compose([
          transforms.ToTensor(),
          ])
    else:
      self.transform = transform

  # Return the number of images in the dataset
  def __len__(self):
    return self.len

  # Get the image and label at a given index
  def __getitem__(self, idx):
    # Return the image and label as a tuple
    """if self.name == 'flickr30k':
        label_idx = idx % self.label_per_img
        img_idx = idx // self.label_per_img
        image = self.transform(self.dataset[img_idx]['img'])
        caption = self.dataset[img_idx]['label'][label_idx]
    else:"""
    image = self.transform(self.dataset[idx]['img'])
    caption = self.dataset[idx]['label']
    return (image, caption)

  # Get the shapes of datas and labels
  def getshapes(self):
    # return shapes of data and labels
    return self.dataset['shape']
  


def find_lr(model, optimiser, loader, timesteps, df, start_val = 1e-6, end_val = 1, beta = 0.99, device=None, show=True):
    n = len(loader) -1
    factor = (end_val / start_val)**(1/n)
    lr = start_val
    optimiser.param_groups[0]['lr'] = lr

    avg_loss, loss = 0., 0.
    lowest_loss = 0
    losses = []
    log_lrs = []

    model = model.to(device=device)
    for i, (x_0, labels) in enumerate(loader, start=1):   # x_0: images
      optimiser.zero_grad()
      x_0 = x_0.to(device)
      # perturb data
      noise = torch.randn_like(x_0).to(device)
      t = torch.randint(1, timesteps + 1, (x_0.shape[0],)).to(device)
      x_t = df.noise_image(x_0, t, noise).to(device)
      t_emb, t_mask = t5_encode_text(labels)
      # use network to recover noise
      pred_noise = model(x_t, t / timesteps, t_emb=t_emb, t_mask=t_mask)

      # loss is measures the element-wise mean squared error between the predicted and true noise
      cost = F.mse_loss(pred_noise, noise)

      loss = beta*loss + (1-beta)*cost.item()
      avg_loss = loss / (1 - beta**i)

      if i > 1 and avg_loss > 4 * lowest_loss:
          print(i, cost.item())
          return log_lrs, losses
      if avg_loss < lowest_loss or i ==  1:
          lowest_loss = avg_loss

      losses.append(avg_loss)
      log_lrs.append(lr)

      cost.backward()
      optimiser.step()

      print(f'Loss: {cost.item():.4f}, lr: {lr:.6f}')
      lr *= factor
      optimiser.param_groups[0]['lr'] = lr

    if show:
      show_lrs(log_lrs, losses)

    return log_lrs, losses



def show_lrs(log_lr, losses):
  _, ax1 = plt.subplots(figsize=(20,10))
  ax1.plot(log_lr, losses)
  ax1.set_xscale('log')
  ax1.set_xticks([1e-3, 2e-3, 1e-2, 1, 10])
  ax1.get_xaxis().get_major_formatter().labelOnlyBase = False
  plt.show()