import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


def loss_gaussian(out, x):
    n = x.shape[1]
    miu = out[:, 0:n]
    std = out[:, n:]
    t1 = (x - miu) ** 2 / (2 * std**2)
    t2 = torch.log(torch.abs(std)+0.0001)
    log_density = t1 + t2

    loss = torch.sum(log_density)
    ttt = loss.cpu().detach().numpy()
    if np.isnan(ttt):
        print('wait')
        print('miu=', miu)
        print('std=', std)
        sys.exit()
    return loss


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_loss(file_name, loss_tr, loss_te, loss_od):
    """
    plot loss from training
    :param file_name:
    :param loss_tr:
    :param loss_te:
    :param loss_od:
    :return:
    """

    plt.figure()
    t = np.arange(len(loss_tr))*10
    plt.plot(t, loss_tr, label='train')
    plt.plot(t, loss_te, label='test ')
    plt.plot(t, loss_od, label='ood  ')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.title('Loss of MADE training')

    img_name = file_name + '.pdf'
    plt.savefig(img_name)
