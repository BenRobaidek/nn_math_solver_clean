import sys
import os
import subprocess
import random
import itertools
from train import train
import torch
import json

rand = True

data_path = '../tencent/data/working/basic/'

mf = (1,2,3)
net_type = ('lstm', 'gru')
epochs = 100,
bs = 8, 16, 64
opt = ('adamax', 'adam', 'sgd')
ly =  (1, 2, 4)
hs = (100, 300, 500, 1000)
num_dir = 2,
embdim = (50, 100, 200, 300, 500, 1000,  1500)
embfix = (False,)#True)
ptemb = (False,)#True)
dropout = (0, .3, .5, .7)
pred_filter = (True, False)

x = list(itertools.product(net_type, epochs, bs, opt, ly, hs, num_dir,
                            embdim, embfix, ptemb, dropout, mf, pred_filter))
if rand: random.shuffle(x)


hyperparam_results = dict()

for (net_type, epoch, bs, opt, ly, hs, num_dir, embdim, embfix, ptemb,
                    dropout, mf, pred_filter) in x:
    if not (embfix and not ptemb):
        hyperparams = {'mf':mf, 'epochs':epoch, 'bs':bs, 'opt':opt,
                    'net_type':net_type, 'ly':ly, 'hs':hs, 'num_dir':num_dir,
                    'embdim':embdim, 'embfix':embfix,
                    'pretrained_emb':ptemb, 'dropout':dropout,
                    'pred_filter':pred_filter}
        results = train(data_path=data_path, train_path='train.tsv',
                val_path='val.tsv', test_path='test.tsv', mf=mf, epochs=5,
                bs=bs, opt=opt, net_type=net_type, ly=ly, hs=hs, num_dir=num_dir,
                emb_dim=embdim, embfix=embfix, pretrained_emb=ptemb, dropout=dropout,
                pred_filter=pred_filter, save_path='./', save=False, verbose=False)
        hyperparam_results[str(hyperparams)] = results
        print(hyperparam_results)
