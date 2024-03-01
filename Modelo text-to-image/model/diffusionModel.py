import numpy as np
import torch
from datetime import datetime
from torchvision import transforms
import matplotlib.pyplot as plt
from model.t5 import t5_encode_text



class DiffusionModel():
  # return beta_sqrt, alpha, alpha_sqrt, gamma, gamma_sqrt for noise and denoise image
  def __init__(self, timesteps, height, beta1=1e-4, beta2=0.02):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    beta = torch.linspace(beta1, beta2, timesteps + 1, device=self.device)
    alpha = 1 - beta
    one_by_sqrt_alpha = 1./alpha.sqrt()
    gamma = torch.cumprod(alpha, axis=0)
    gamma[0] = 1
    sqrt_one_minus_gamma = (1. - gamma).sqrt()
    beta_by_sqrt_one_minus_gamma = beta/sqrt_one_minus_gamma
    self.noise_schedule_dict = {'alpha':alpha, 'sqrt_alpha':alpha.sqrt(),
                   'beta':beta, 'sqrt_beta':beta.sqrt(),
                   'gamma':gamma, 'sqrt_gamma':gamma.sqrt(),
                   'one_by_sqrt_alpha':one_by_sqrt_alpha,
                   'sqrt_one_minus_gamma':sqrt_one_minus_gamma,
                   'beta_by_sqrt_one_minus_gamma':beta_by_sqrt_one_minus_gamma}
    self.timesteps = timesteps
    self.height = height

  def show_noise_schedule(self):
    rows = 5
    cols = 2
    plt.figure(figsize=(16,16))
    for i, schedule in enumerate(self.noise_schedule_dict):
      plt.subplot(rows, cols, i + 1)
      plt.title(schedule)
      curr_schedule = self.noise_schedule_dict[schedule]
      plt.plot(curr_schedule.to('cpu'))

  def show_noise_schedule2(self):
    for i, schedule in enumerate(self.noise_schedule_dict):
      plt.figure(figsize=(4,4))
      plt.title(schedule)
      curr_schedule = self.noise_schedule_dict[schedule]
      plt.plot(curr_schedule.to('cpu'))

  def noise_image(self, x_0, time, noise=None):
    if noise is None:
      noise = torch.randn_like(x_0)
    img = self.noise_schedule_dict['sqrt_gamma'].to(self.device)[time, None, None, None] * x_0.to(self.device)+(self.noise_schedule_dict['sqrt_one_minus_gamma'].to(self.device)[time, None, None, None]) * noise.to(self.device)
    return img

  def unorm(self, img):
    #img = torch.tensor(img)
    #img = torch.clamp(img, -1.,1.)
    #min, max = torch.aminmax(img)
    #img = (img - min)/(max-min)
    std = 0.2753
    mean = 0.6926
    img = (img*std)+mean
    #img = torch.tensor(img)
    #min, max = torch.aminmax(img)
    #img = (img - min)/(max-min)
    return img

  def norm(self, img):
    img = torch.tensor(img)
    img = torch.clamp(img, -1.,1.)
    min, max = torch.aminmax(img)
    img = (img - min)/(max-min)
    return img

  def simulate_forward_diffusion(self, time_step, dataset, image=None, num_images=10):
    plt.figure(figsize=(num_images*2, 2))
    if image is None:
      idx = np.random.randint(len(dataset))
      image, curr_label = dataset[idx]
    stepsize = int(time_step/num_images)
    steps = range(0, time_step, stepsize)
    for i, curr_time in enumerate(steps):
      time = torch.Tensor([curr_time]).type(torch.int64)
      img = self.noise_image(image, time).to('cpu')
      img = self.unorm(img)
      plt.subplot(1, num_images + 1, i + 1)
      plt.axis('off')
      plt.title("time: {}".format(curr_time))
      plt.imshow(img[0].permute(1,2,0))

  def denoise_add_noise(self, x_t, t, pred_noise, z=None):
    if z is None:
      z = torch.randn_like(x_t)
    sqrt_beta = self.noise_schedule_dict['sqrt_beta'][t, None, None, None].to(self.device)
    noise = z.to(self.device) * sqrt_beta
    beta_by_sqrt_one_minus_gamma = self.noise_schedule_dict['beta_by_sqrt_one_minus_gamma'][t, None, None, None].to(self.device)
    pred_noise = pred_noise.to(self.device) * beta_by_sqrt_one_minus_gamma
    mean = (x_t - pred_noise) * self.noise_schedule_dict['one_by_sqrt_alpha'][t, None, None, None].to(self.device)
    if t > 1:
      x_t_minus_1 = mean + noise
    else:
      x_t_minus_1 = mean
    return x_t_minus_1

  # sample with context using standard algorithm
  @torch.no_grad()
  def sample_ddpm_context(self, model, n_sample, labels, text_emb_func=None, save_rate=20, samples=None, timesteps=None):
    # x_T
    if samples is None:
        samples = torch.randn(size=(n_sample, 3, self.height, self.height)).to(self.device)
    if timesteps is None:
        timesteps = self.timesteps
    # arrays to keep track of generated steps for plotting
    self.intermediate = []
    self.intermediate_time = []
    for i in range(timesteps, 0, -1):
        # reshape time tensor
        t = torch.tensor([i/timesteps],).to(self.device)
        t = t*torch.ones(n_sample).to(self.device)
        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else torch.zeros_like(samples)
        if text_emb_func != None:
            t_emb, t_mask = text_emb_func(labels)
        else:
            t_emb, t_mask = t5_encode_text(labels)
        eps = model(samples, t, t_emb=t_emb, t_mask=t_mask)    # predict noise e_(x_t,t, ctx)
        samples = self.denoise_add_noise(samples, i, eps, z=z)
        if i % save_rate==0 or i==timesteps or i<8:
            print(f'EPS -> Mean: {eps.mean()}, std: {eps.std()}')
            print(f'SAMPLES -> Mean: {samples.mean()}, std: {samples.std()}')
            self.intermediate.append(samples.detach().cpu().numpy())
            self.intermediate_time.append(i)
    self.intermediate = np.stack(self.intermediate)
    return samples

  def draw_samples_process(self, path=""):
    cols = len(self.intermediate)
    rows = self.intermediate[0].shape[0]
    plt.figure(figsize=(cols*2, rows*2))
    for i, curr_imgs in enumerate(self.intermediate):
      curr_time = self.intermediate_time[i]
      for j, curr_img in enumerate(curr_imgs):
        curr_img = self.unorm(curr_img)
        curr_img = np.transpose(curr_img, (1, 2, 0))
        plt.subplot(rows, cols, (j*cols) + i + 1)
        plt.axis('off')
        plt.title("time: {}".format(curr_time))
        #plt.imshow(curr_img.permute(1,2,0))
        plt.imshow(curr_img)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{path}samples_process_{timestamp}.png')
    #plt.close()