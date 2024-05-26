from model.dataset import *
from torch.utils.data import DataLoader
from model.t5 import t5_encode_text
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './weights/'


def train(model, df, optimizer, dataset, n_epoch=100, batch_size=32, device=DEVICE, save_dir=SAVE_DIR):
    log_file='./metrics/text2img_progress.txt'
    def show_msg(msg, file=log_file):
        if file is not None:
            with open(file, 'a') as f:
                f.write(msg+"\n")
        print(msg)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    open(log_file, 'w').close()
    list_total_loss = []
    training_steps = 0
    model.train()
    mean_loss_train = []
    for epoch in range(n_epoch):
        show_msg("------------------------------------ epoch {:03d} ({} steps) ------------------------------------".format(epoch + 1, training_steps))
        total_loss = 0
        loss_list = []
        # linearly decay of learning rate
        #optimizer.param_groups[0]['lr'] = learning_rate*(1-(epoch/n_epoch))
        for x_0, labels in dataloader:   # x_0: images
            optimizer.zero_grad()
            x_0 = x_0.to(device)
            # perturb data
            noise = torch.randn_like(x_0).to(device)
            t = torch.randint(1, df.timesteps + 1, (x_0.shape[0],)).to(device)
            x_t = df.noise_image(x_0, t, noise).to(device)
            t_emb, t_mask = t5_encode_text(labels)
            # use network to recover noise
            pred_noise = model(x_t, t/df.timesteps, t_emb=t_emb, t_mask=t_mask)

            # loss is measures the element-wise mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_noise, noise)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            training_steps+=1
            #if (training_steps%100) == 0:
                #print("Total train step: {}, Loss: {}".format(training_steps,loss))
        loss_list = np.array(loss_list)
        mean_loss_train.append(loss_list.mean())
        total_loss = loss_list.sum()
        show_msg("Mean loss: {}".format(loss_list.mean()))
        list_total_loss.append(total_loss)
        #valid
        
    # save model periodically
        if epoch%10==0 or epoch == int(n_epoch-1):
            torch.save(model.state_dict(), save_dir + "model_superres_{:03d}.pth".format(epoch + 1))
            show_msg('saved model at ' + save_dir + "model_superres_{:03d}.pth".format(epoch + 1))
            plt.figure()
            plt.plot(mean_loss_train)
            plt.title("Train Loss per Epoch")
            plt.savefig('train_text2image.png')
            plt.close()
        
    plt.figure()
    plt.plot(list_total_loss)
    plt.title("Total Loss vs Epoch")
    plt.savefig('train.png')

    show_msg("Fin del entrenamiento")