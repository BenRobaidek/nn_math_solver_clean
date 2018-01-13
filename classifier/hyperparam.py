import sys
import os
import subprocess
import random
import itertools
from train import train
import torch

rand = True

mf = (1,2,3)
net_type = ('lstm', 'gru')
epochs = 100,
bs = 8, 16, 64
opt = ('adamax', 'adam', 'sgd')
num_lay =  (1, 2, 4)
hs = (100, 300, 500, 1000)
num_dir = 2,
embdim = (50, 100, 200, 300, 500, 1000,  1500)
embfix = (False,)#True)
ptemb = (False,)#True)
dropout = (0, .3, .5, .7)

x = list(itertools.product(net_type, epochs, bs, opt, num_lay, hs, num_dir,
                                            embdim, embfix, ptemb, dropout, mf))
if rand: random.shuffle(x)

for (net_type, epoch, bs, opt, num_lay, hs, num_dir, embdim, embfix, ptemb,
                                                        dropout, mf) in x:
    if not (embfix and not ptemb):
        train(data_path='../tencent/data/working/basic/', train_path='train.tsv',
                val_path='val.tsv', test_path='test.tsv', mf=1, epochs=5,
                bs=8, opt='adam', net_type='lstm', ly=1, hs=100, num_dir=1,
                emb_dim=100, embfix=False, pretrained_emb=False, dropout=0.0,
                pred_filter=True, save_path='./', save=False, verbose=False)
