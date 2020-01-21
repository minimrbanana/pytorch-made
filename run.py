"""
Trains MADE on Binarized MNIST, which can be downloaded here:
https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import torch.distributions as tdist
import time
import logging

from made import MADE
from utils import loss_gaussian, check_path
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# ------------------------------------------------------------------------------


def run_epoch(split, upto=None):
    torch.set_grad_enabled(split=='train')  # enable/disable grad for efficiency of forwarding test batches
    model.train() if split == 'train' else model.eval()
    nsamples = 1 if split == 'train' else args.samples
    if split == 'train':
        x = xtr
    elif split == 'test':
        x = xte
    else:
        x = xod
    N,D = x.size()
    B = 128  # batch size
    nsteps = N//B if upto is None else min(N//B, upto)
    lossfs = []
    for step in range(nsteps):
        
        # fetch the next batch of data
        xb = Variable(x[step*B:step*B+B])
        
        # get the logits, potentially run the same batch a number of times, resampling each time
        # xbhat = torch.zeros_like(xb)
        xbhat = torch.zeros_like(torch.cat((xb, xb), 1))
        for s in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            if step % args.resample_every == 0 or split == 'test' or split == 'ood':  # if in test, cycle masks every time
                model.update_masks()
            # forward the model
            xbhat += model(xb)
        xbhat /= nsamples

        # view as numpy array
        # tt = xbhat.detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(tt[0, :].reshape(28, 28))
        # plt.show()
        
        # evaluate the binary cross entropy loss
        # loss = F.binary_cross_entropy_with_logits(xbhat, xb, size_average=False) / B
        if step==208:
            print("ggg")
        loss = loss_gaussian(xbhat, xb) / B
        lossf = loss.data.item()
        if np.isnan(lossf):
            print('wait')
        lossfs.append(lossf)
        
        # backward/update
        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()
    if epoch % 5==0:
        print("%s epoch average loss: %f" % (split, np.mean(lossfs)))
    if epoch == 300:
        log_msg = str(np.mean(lossfs))
        logger.info(log_msg)
    #     print('200')
# ------------------------------------------------------------------------------


def init_log(ARGS):
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # Creating log file
    save_path = './mnist/'
    check_path(save_path)
    # path_base = '/media/yu/data/yu/code/gp_whittle/WhittleNetwork/dev/'
    file_base = 'made_' + str(ARGS.hiddens) + '_on_mnist_'
    logging.basicConfig(
        filename=save_path + file_base + current_time + '.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    return logger


# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', default='/media/yu/data/yu/dataset/binary_mnist/binarized_mnist.npz', type=str, help="Path to binarized_mnist.npz")
    parser.add_argument('-q', '--hiddens', type=str, default='1000,1000,1000', help="Comma separated sizes for hidden layers, e.g. 500, or 500,500")
    parser.add_argument('-n', '--num-masks', type=int, default=1, help="Number of orderings for order/connection-agnostic training")
    parser.add_argument('-r', '--resample-every', type=int, default=20, help="For efficiency we can choose to resample orders/masks only once every this many steps")
    parser.add_argument('-s', '--samples', type=int, default=1, help="How many samples of connectivity/masks to average logits over during inference")
    args = parser.parse_args()
    # --------------------------------------------------------------------------
    
    # reproducibility is good
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # init logger
    logger = init_log(args)

    # load the dataset
    # print("loading binarized mnist from", args.data_path)
    # mnist = np.load(args.data_path)
    # xtr, xte = mnist['train_data'], mnist['valid_data']
    # from mlxtend.data import loadlocal_mnist
    # xtr, _ = loadlocal_mnist(
    #     images_path='/media/yu/data/yu/dataset/mnist/train-images-idx3-ubyte',
    #     labels_path='/media/yu/data/yu/dataset/mnist/train-labels-idx1-ubyte')
    # xte, _ = loadlocal_mnist(
    #     images_path='/media/yu/data/yu/dataset/mnist/t10k-images-idx3-ubyte',
    #     labels_path='/media/yu/data/yu/dataset/mnist/t10k-labels-idx1-ubyte')

    data_train = np.fromfile('/media/yu/data/yu/code/gp_whittle/WhittleNetwork/train_mnist.dat',
                             dtype=np.float64).reshape(-1, 224)
    data_pos = np.fromfile('/media/yu/data/yu/code/gp_whittle/WhittleNetwork/test_mnist_positive.dat',
                           dtype=np.float64).reshape(-1, 224)
    data_neg = np.fromfile('/media/yu/data/yu/code/gp_whittle/WhittleNetwork/test_mnist_negative.dat',
                           dtype=np.float64).reshape(-1, 224)
    n_RV = 224  # number of RVs
    scope_list = np.arange(n_RV)
    scope_temp = np.delete(scope_list, np.where(scope_list % 16 == 8))
    init_scope = list(np.delete(scope_temp, np.where(scope_temp % 16 == 15)))
    # modify data to remove 0 (imag) columns
    data_train = data_train[:, init_scope]
    data_pos = data_pos[:, init_scope]
    data_neg = data_neg[:, init_scope]

    xtr = torch.from_numpy(data_train).float().cuda()
    xte = torch.from_numpy(data_pos).float().cuda()
    xod = torch.from_numpy(data_neg).float().cuda()

    # construct model and ship to GPU
    hidden_list = list(map(int, args.hiddens.split(',')))
    model = MADE(xtr.size(1), hidden_list, xtr.size(1)*2, num_masks=args.num_masks)
    print("number of model parameters:",sum([np.prod(p.size()) for p in model.parameters()]))
    model.cuda()

    for step_size in [1e-5]:
        for decay in [1e-5]:

            # set up the optimizer
            # opt = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
            opt = torch.optim.Adam(model.parameters(), step_size, weight_decay=decay)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)

            log_msg = 'step size = ' + str(step_size) + ', decay = ' + str(decay)
            logger.info(log_msg)

            # start the training
            for epoch in range(301):
                if epoch%10==0:
                    print("epoch %d" % (epoch, ))
                scheduler.step(epoch)
                run_epoch('train')
                run_epoch('test')  # run validation, which is pos
                run_epoch('ood')  # run test, which is ood

            print("optimization done. full test set eval:")
            # run_epoch('test')

