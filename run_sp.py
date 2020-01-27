"""
Trains MADE on my S&P data
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

from made import MADE
from utils import check_path, plot_loss, loss_gaussian
import os
import logging
import setproctitle

# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# ------------------------------------------------------------------------------


# path settings
# my machine
path_base = '/media/yu/data/yu/code/gp_whittle/WhittleNetwork/'
if not os.path.exists(path_base):
    # dgx1 machine
    path_base = "/dataset/whittle_data/"


def run_epoch(split, upto=None):
    proc_name = "Yu-MADE:%d:%d-sp-lr%.8f" % (epoch, args.epoch, args.learning_rate)
    setproctitle.setproctitle(proc_name)
    torch.set_grad_enabled(split == 'train')  # enable/disable grad for efficiency of forwarding test batches
    model.train() if split == 'train' else model.eval()
    nsamples = 1 if split == 'train' else args.samples
    if split == 'train':
        x = xtr
        log_label = 'train'
    elif split == 'test':
        x = xte
        log_label = 'test '
    else:
        x = xod
        log_label = 'ood  '
    N, D = x.size()
    B = args.batch_size  # batch size, 4
    nsteps = N // B if upto is None else min(N // B, upto)
    lossfs = []
    for step in range(nsteps):

        # fetch the next batch of data
        xb = Variable(x[step * B:step * B + B])

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

        # evaluate the binary cross entropy loss
        # loss = F.binary_cross_entropy_with_logits(xbhat, xb, size_average=False) / B
        loss = loss_gaussian(xbhat, xb) / B
        lossf = loss.data.item()
        lossfs.append(lossf)

        # backward/update
        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()
    if epoch % 10 == 0:
        log_msg = "%s epoch %d average loss: %f" % (log_label, epoch, np.mean(lossfs))
        print(log_msg)
        logger.info(log_msg)

    return np.mean(lossfs)


def init_log(args):
    import time
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # Creating log file
    save_path = './sp/'
    check_path(save_path)
    file = save_path + 'q_' + args.hiddens + '_n_' + str(args.num_masks) + \
           '_lr_' + str(args.learning_rate) + '_wd_' + str(args.weight_decay) + \
           '_b_' + str(args.batch_size) + '_' + current_time
    logging.basicConfig(
        filename=file + '.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    return logger, file

# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', default='/media/yu/data/yu/dataset/binary_mnist/binarized_mnist.npz',
                        type=str, help="Path to binarized_mnist.npz")
    parser.add_argument('-q', '--hiddens', type=str, default='1000',
                        help="Comma separated sizes for hidden layers, e.g. 500, or 500,500")
    parser.add_argument('-n', '--num-masks', type=int, default=1,
                        help="Number of orderings for order/connection-agnostic training")
    parser.add_argument('-r', '--resample-every', type=int, default=20,
                        help="For efficiency we can choose to resample orders/masks only once every this many steps")
    parser.add_argument('-s', '--samples', type=int, default=1,
                        help="How many samples of connectivity/masks to average logits over during inference")
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.02,
                        help="Learning rate")
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument('-b', '--batch-size', type=int, default=8,
                        help="Batch size")
    parser.add_argument('-ep', '--epoch', type=int, default=300,
                        help="number of epochs")
    args = parser.parse_args()
    # --------------------------------------------------------------------------

    # reproducibility is good
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # init logger
    logger, file_name = init_log(args)
    log_msg = '\n--hiddens=' + str(args.hiddens) + \
              '\n--num-masks=' + str(args.num_masks) + \
              '\n--learning-rate=' + str(args.learning_rate) + \
              '\n--weight-decay=' + str(args.weight_decay) + \
              '\n--batch-size=' + str(args.batch_size)
    print(log_msg)
    logger.info(log_msg)

    data_train = np.fromfile(path_base + 'train_SP.dat',
                             dtype=np.float64).reshape(-1, 374)
    data_pos = data_train.copy()
    data_neg = data_train.copy()
    n_RV = 374  # number of RVs
    scope_list = np.arange(n_RV)
    scope_temp = np.delete(scope_list, np.where(scope_list % 34 == 17))
    init_scope = list(np.delete(scope_temp, np.where(scope_temp % 34 == 33)))
    # modify data to remove 0 (imag) columns
    data_train = data_train[:, init_scope]
    data_pos = data_pos[:, init_scope]
    data_neg = data_neg[:, init_scope]

    xtr = torch.from_numpy(data_train).float().cuda()
    xte = torch.from_numpy(data_pos).float().cuda()
    xod = torch.from_numpy(data_neg).float().cuda()

    # construct model and ship to GPU
    hidden_list = list(map(int, args.hiddens.split(',')))
    model = MADE(xtr.size(1), hidden_list, xtr.size(1) * 2, num_masks=args.num_masks)
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    model.cuda()

    # set up the optimizer
    opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)

    # list to store loss
    loss_tr = []
    loss_te = []
    loss_od = []
    # start the training
    for epoch in range(args.epoch):
        scheduler.step(epoch)
        loss_tr.append(run_epoch('train'))
        loss_te.append(run_epoch('test'))  # run validation, which is pos
        loss_od.append(run_epoch('ood'))  # run test, which is ood

    print("optimization done")
    plot_loss(file_name, loss_tr, loss_te, loss_od)
    # run_epoch('test')

