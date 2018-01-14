import sys
import os
import subprocess
import random
import itertools
from train import train
import torch
import json
import numpy as np

def main():

    # LOAD FROM CONFIG FILE
    with open('../tests/exp1/config.json', 'r') as f:
        config = json.load(f)

    x = list(itertools.product(config['net_type'], config['epochs'], config['bs'], config['opt'], config['ly'], config['hs'], config['num_dir'], config['embdim'], config['embfix'], config['ptemb'], config['dropout', config['mf'], config['pred_filter']))
    if rand: random.shuffle(x)


    hyperparam_results = dict()

    # HYPERPARAM SEARCH
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
            results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
            hyperparam_results[str(hyperparams)] = results

    print(hyperparam_results)

    # RETRAIN/SAVE BEST MODEL

if __name__ == '__main__':
    main()
