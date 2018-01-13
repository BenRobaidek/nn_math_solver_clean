import sys
import os
import subprocess
import random
import itertools
from train import train
import torch
import json
import numpy as np

rand = True

data_path = '../tencent/data/working/basic/'

mf = (1,)
net_type = ('lstm',)
epochs = 3,
bs = 8,
opt = ('adamax',)
ly =  1, 2
hs = (100,)
num_dir = 1,
embdim = (50,)
embfix = (False,)#True)
ptemb = (False,)#True)
dropout = (0,)
pred_filter = (True,)

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
                val_path='val.tsv', test_path='test.tsv', mf=mf, epochs=epoch,
                bs=bs, opt=opt, net_type=net_type, ly=ly, hs=hs, num_dir=num_dir,
                emb_dim=embdim, embfix=embfix, pretrained_emb=ptemb, dropout=dropout,
                pred_filter=pred_filter, save_path='./', save=False, verbose=False)
        hyperparam_results[str(hyperparams)] = results
        hyperparam_results = np.sort(hyperparam_results, key=lambda x : x['accuracy'])
        print(hyperparam_results)
