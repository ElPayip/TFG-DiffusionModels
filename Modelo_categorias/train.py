import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './weights/'
# diffusion hyperparameters
TIMESTEPS = 500

def train(model, diffusion, optimizer, train_dataset, validation_dataset=None, batch_size=128, epochs=100, device=DEVICE):
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    if validation_dataset:
        dataloader_valid = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

    training_steps = 0
    mean_loss_train = []
    mean_loss_valid = []
    for epoch in range(epochs):
        print("------ epoch {:03d} ------".format(epoch + 1))
        # linearly decay learning rate
        #optimizer.param_groups[0]['lr'] = learning_rate*(1-(epoch/n_epoch))
        total_loss = []
        model.train()
        for x_0, labels in dataloader_train:   # x_0: images
            optimizer.zero_grad()
            x_0 = x_0.to(device)
            # perturb data
            noise = torch.randn_like(x_0).to(device)
            t = torch.randint(1, diffusion.timesteps + 1, (x_0.shape[0],)).to(device)
            x_t = diffusion.noise_image(x_0, t, noise)
            # use network to recover noise
            pred_noise = model(x_t, t / diffusion.timesteps, c=labels.float().to(device))

            # loss is measures the element-wise mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            training_steps+=1
            # if (training_steps%100) == 0:
            #     print("Total train step: {}, Loss: {}".format(training_steps,loss))
        mean_loss_train.append(np.mean(total_loss))

        print("Total Loss: {}".format(mean_loss_train[-1]))
        # save model periodically
        if epoch%5==0 or epoch == int(epochs-1):
            torch.save(model.state_dict(), SAVE_DIR + "model_{:03d}.pth".format(epoch + 1))
            print('saved model at ' + SAVE_DIR + "model_{:03d}.pth".format(epoch + 1))
        
        if dataloader_valid:
            model.eval()
            total_loss = []
            with torch.no_grad():
                for x_0, labels in dataloader_valid:
                    x_0 = x_0.to(device)
                    noise = torch.randn_like(x_0).to(device)
                    t = torch.randint(1, diffusion.timesteps + 1, (x_0.shape[0],)).to(device)
                    x_t = diffusion.noise_image(x_0, t, noise)
                    # use network to recover noise
                    pred_noise = model(x_t, t / diffusion.timesteps, c=labels.float().to(device))

                    # loss is measures the element-wise mean squared error between the predicted and true noise
                    loss = F.mse_loss(pred_noise, noise)
                    total_loss.append(loss.item())
                mean_loss_valid.append(np.mean(total_loss))
    model.eval()