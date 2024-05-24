import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms.functional import resize

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './weights/'
METRICS_DIR = './metrics/'
# diffusion hyperparameters
TIMESTEPS = 1000

HEIGHT = 128 # 64x64 image
LOW_HEIGHT = 64

def train(model, diffusion, optimizer, train_dataset, validation_dataset=None, batch_size=32, epochs=100, device=DEVICE):
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    if validation_dataset:
        dataloader_val = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)

    log_file=METRICS_DIR+'training_progress_superres.txt'
    def show_msg(msg, file=log_file):
        if file is not None:
            with open(file, 'a') as f:
                f.write(msg+"\n")
        print(msg)
    open(log_file, 'w').close()

    lr = optimizer.param_groups[0]['lr']
    training_steps = 0
    list_total_loss = []
    list_total_loss_val = []
    for epoch in range(epochs):
        model.train()
        show_msg("------------------------------------ epoch {:03d} ({} steps) ------------------------------------".format(epoch + 1, training_steps))
        # linearly decay learning rate
        optimizer.param_groups[0]['lr'] = lr*(1-(epoch/epochs))
        total_loss = 0
        loss_list = []
        for x_0 in dataloader_train:   # x_0: images
            optimizer.zero_grad()
            x_0 = x_0.to(device)
            x_lowres = resize(x_0, (LOW_HEIGHT,LOW_HEIGHT))
            x_lowres = resize(x_lowres, (HEIGHT, HEIGHT)).to(device)
            # perturb data
            noise = torch.randn_like(x_0).to(device)
            t = torch.randint(1, TIMESTEPS + 1, (x_0.shape[0],)).to(device)
            x_t = diffusion.noise_image(x_0, t, noise)
            # use network to recover noise
            pred_noise = model(x_t, t / TIMESTEPS, x_lowres)

            # loss is measures the element-wise mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            training_steps+=1
            #if (training_steps%100) == 0:
            #    show_msg("Total train step: {}, Loss: {}".format(training_steps,loss))
        loss_list_ = np.array(loss_list)
        show_msg("Max loss: {}".format(loss_list_.max()))
        show_msg("Min loss: {}".format(loss_list_.min()))
        total_loss = loss_list_.sum()
        list_total_loss.append(loss_list_.mean())
        show_msg("Mean loss: {}".format(loss_list_.mean()))
        show_msg("Std loss: {}".format(loss_list_.std()))
        show_msg("Total Loss: {}".format(total_loss))
        
        loss_val_list = []
        model.eval()
        with torch.no_grad():
            for x_0 in dataloader_val:
                x_0 = x_0.to(device)
                x_lowres = resize(x_0, (LOW_HEIGHT,LOW_HEIGHT))
                x_lowres = resize(x_lowres, (HEIGHT, HEIGHT)).to(device)
                # perturb data
                noise = torch.randn_like(x_0).to(device)
                t = torch.randint(1, TIMESTEPS + 1, (x_0.shape[0],)).to(device)
                x_t = diffusion.noise_image(x_0, t, noise)
                # use network to recover noise
                pred_noise = model(x_t, t / TIMESTEPS, x_lowres)

                # loss is measures the element-wise mean squared error between the predicted and true noise
                loss_val = F.mse_loss(pred_noise, noise)
                loss_val_list.append(loss_val.item())
        val_mean = np.array(loss_val_list).mean()
        list_total_loss_val.append(val_mean)
        show_msg("Validation Mean Loss: {}".format(val_mean))
        
        # save model periodically
        if epoch%5==0 or epoch == int(epochs-1):
            torch.save(model.state_dict(), SAVE_DIR + "model_superres_{:03d}.pth".format(epoch + 1))
            show_msg('saved model at ' + SAVE_DIR + "model_superres_{:03d}.pth".format(epoch + 1))
            
        metrics = dict()
        metrics['train'] = list_total_loss
        metrics['validation'] = list_total_loss_val
        np.save(f'{METRICS_DIR}superres.npy', metrics)