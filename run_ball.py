"""
Trains MADE on Binarized MNIST, which can be downloaded here:
https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz
"""
import argparse
import pickle

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
    proc_name = "Yu-MADE:%d:%d-billiards-lr%.8f" % (epoch, args.epoch, args.learning_rate)
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
    B = args.batch_size  # batch size
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


# ------------------------------------------------------------------------------


def init_log(args):
    import time
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # Creating log file
    save_path = './billiards/'
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
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-2,
                        help="Learning rate")
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help="Batch size")
    parser.add_argument('-ep', '--epoch', type=int, default=30,
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

    data_path = path_base + '/datasets/billiards_data/'
    # Load training data
    data = pickle.load(open(data_path + 'billiards_train_10000.pkl', 'rb'))
    # extract data and do DTFT
    positions = data['y']
    positions = positions[..., :2]
    positions = positions[0:9700, ...]
    # normalize to [1.2, 8.8]
    # positions = 7.6/(np.max(positions)-np.min(positions))*(positions-np.min(positions))+1.2
    # normalize to [-1, 1]
    data_max = np.max(positions)
    data_min = np.min(positions)
    positions = 2 / (data_max - data_min) * (positions - data_min) - 1
    # positions =
    data_rfft = np.fft.rfft(positions, axis=1)
    d_r = data_rfft.real
    d_i = data_rfft.imag
    data_x1 = np.concatenate([d_r[:, :, 0, 0], d_i[:, :, 0, 0]], axis=1)
    data_y1 = np.concatenate([d_r[:, :, 0, 1], d_i[:, :, 0, 1]], axis=1)
    data_x2 = np.concatenate([d_r[:, :, 1, 0], d_i[:, :, 1, 0]], axis=1)
    data_y2 = np.concatenate([d_r[:, :, 1, 1], d_i[:, :, 1, 1]], axis=1)
    data_x3 = np.concatenate([d_r[:, :, 2, 0], d_i[:, :, 2, 0]], axis=1)
    data_y3 = np.concatenate([d_r[:, :, 2, 1], d_i[:, :, 2, 1]], axis=1)
    # data_train in the form of [x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, x3r, x3i, y3r, y3i]
    data_train = np.concatenate((data_x1, data_y1, data_x2, data_y2, data_x3, data_y3), axis=1)

    # Load test data
    # data = pickle.load(open(data_path + 'billiards_test_gravity.pkl', 'rb'))
    # data = scipy.io.loadmat(data_path + 'billiards_test.mat')
    # extract data and do DTFT
    positions = data['y']
    positions = positions[..., :2]
    positions = positions[9700:, ...]
    # normalize to [1.2, 8.8]
    # positions = 7.6/(np.max(positions)-np.min(positions))*(positions-np.min(positions))+1.2
    # normalize to [-1, 1]
    positions = 2 / (data_max - data_min) * (positions - data_min) - 1
    data_rfft = np.fft.rfft(positions, axis=1)
    d_r = data_rfft.real
    d_i = data_rfft.imag
    data_x1 = np.concatenate([d_r[:, :, 0, 0], d_i[:, :, 0, 0]], axis=1)
    data_y1 = np.concatenate([d_r[:, :, 0, 1], d_i[:, :, 0, 1]], axis=1)
    data_x2 = np.concatenate([d_r[:, :, 1, 0], d_i[:, :, 1, 0]], axis=1)
    data_y2 = np.concatenate([d_r[:, :, 1, 1], d_i[:, :, 1, 1]], axis=1)
    data_x3 = np.concatenate([d_r[:, :, 2, 0], d_i[:, :, 2, 0]], axis=1)
    data_y3 = np.concatenate([d_r[:, :, 2, 1], d_i[:, :, 2, 1]], axis=1)
    # data_train in the form of [x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, x3r, x3i, y3r, y3i]
    data_pos = np.concatenate((data_x1, data_y1, data_x2, data_y2, data_x3, data_y3), axis=1)

    # Load outlier data
    data = pickle.load(open(data_path + 'billiards_test_drift.pkl', 'rb'))
    # extract data and do DTFT
    positions = data['y']
    positions = positions[..., :2]
    # normalize to [1.2, 8.8]
    # positions = 7.6/(np.max(positions)-np.min(positions))*(positions-np.min(positions))+1.2
    # normalize to [-1, 1]
    positions = 2 / (data_max - data_min) * (positions - data_min) - 1
    ##### create simple outlier
    np.random.seed(20191120)
    states00 = positions.copy()
    for i in range(positions.shape[0]):
        for j in range(3):
            # choose x or y
            xy = np.random.randint(0, 2)
            # set it constant
            rand_pos = np.random.rand(1) * 2 - 1 + np.random.rand(100) * 0.2
            rand_pos[rand_pos > 1] = 1
            rand_pos[rand_pos < -1] = -1
            states00[i, :, j, xy] = rand_pos
    positions = states00.copy()
    #####

    data_rfft = np.fft.rfft(positions, axis=1)
    d_r = data_rfft.real
    d_i = data_rfft.imag
    data_x1 = np.concatenate([d_r[:, :, 0, 0], d_i[:, :, 0, 0]], axis=1)
    data_y1 = np.concatenate([d_r[:, :, 0, 1], d_i[:, :, 0, 1]], axis=1)
    data_x2 = np.concatenate([d_r[:, :, 1, 0], d_i[:, :, 1, 0]], axis=1)
    data_y2 = np.concatenate([d_r[:, :, 1, 1], d_i[:, :, 1, 1]], axis=1)
    data_x3 = np.concatenate([d_r[:, :, 2, 0], d_i[:, :, 2, 0]], axis=1)
    data_y3 = np.concatenate([d_r[:, :, 2, 1], d_i[:, :, 2, 1]], axis=1)
    # data_train in the form of [x1r, x1i, y1r, y1i, x2r, x2i, y2r, y2i, x3r, x3i, y3r, y3i]
    data_neg = np.concatenate((data_x1, data_y1, data_x2, data_y2, data_x3, data_y3), axis=1)

    n_RV = 612  # number of RVs
    p = 6  # dim
    L = 100  # TS length
    scope_list = np.arange(n_RV)
    # 1. standard
    # scope_temp = np.delete(scope_list, np.where(scope_list % 102 == 51))
    # init_scope = list(np.delete(scope_temp, np.where(scope_temp % 102 == 101)))
    # 2. removing high frequencies
    scope_list_x1r = np.arange(29)
    scope_list_x1i = np.arange(52, 81)
    scope_list_x1 = np.concatenate([scope_list_x1r, scope_list_x1i])
    scope_list_y1 = scope_list_x1 + 102
    scope_list_1 = np.concatenate([scope_list_x1, scope_list_y1])
    scope_list_2 = scope_list_1 + 204
    scope_list_3 = scope_list_2 + 204
    scope_list = np.concatenate([scope_list_1, scope_list_2, scope_list_3])
    init_scope = list(scope_list)
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

