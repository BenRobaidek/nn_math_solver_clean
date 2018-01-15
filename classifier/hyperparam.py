import sys
import os
import argparse
import subprocess
import random
import itertools
from train import train
import torch
import json
import numpy as np

def main():
    # PARSE ARGS
    parser = argparse.ArgumentParser(description='LSTM text classifier')
    parser.add_argument('-config', type=str,
        default='../tests/exp1/config.json',
        help='config file path [default: ../tests/exp1/config.json]')
    parser.add_argument('-hyperparam_results', type=str,
        default='../tests/exp1/hyperparam_results.json',
        help='hyperparameter search results file path' \
            '[default: ../tests/exp1/hyperparam_results.json]')
    args = parser.parse_args()

    # LOAD FROM CONFIG FILE
    print('Loading parameters from config file:', args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)

    print('CONFIG:', config)

    x = list(itertools.product(config['net_type'], config['epochs'],
        config['bs'], config['opt'], config['ly'], config['hs'],
        config['num_dir'], config['embdim'], config['embfix'], config['ptemb'],
        config['dropout'], config['mf'], config['pred_filter']))
    if bool(config['rand']): random.shuffle(x)

    # HYPERPARAM SEARCH
    hyperparam_results = None
    try:
        hyperparam_results = json.load(args.hyperparam_results)
    except FileNotFoundError:
        print('exception')
    hyperparam_results = dict()
    for (net_type, epoch, bs, opt, ly, hs, num_dir, embdim, embfix, ptemb,
                        dropout, mf, pred_filter) in x:
        if not (embfix and not ptemb):
            hyperparams = {'mf':mf, 'epochs':epoch, 'bs':bs, 'opt':opt,
                        'net_type':net_type, 'ly':ly, 'hs':hs, 'num_dir':num_dir,
                        'embdim':embdim, 'embfix':embfix,
                        'pretrained_emb':ptemb, 'dropout':dropout,
                        'pred_filter':pred_filter}
            results = None
            print(hyperparams)
            try:
                results = train(data_path=config['data_path'], train_path='train.tsv',
                        val_path='val.tsv', test_path='test.tsv', mf=mf, epochs=epoch,
                        bs=bs, opt=opt, net_type=net_type, ly=ly, hs=hs, num_dir=num_dir,
                        emb_dim=embdim, embfix=bool(embfix), pretrained_emb=bool(ptemb), dropout=dropout,
                        pred_filter=bool(pred_filter), save_path='./', save=False, verbose=False)
                results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
            except RuntimeError:
                print('Oops... Ran out of memory')
            hyperparam_results[str(hyperparams)] = results

    #print(hyperparam_results)

    # RETRAIN/SAVE BEST MODEL

if __name__ == '__main__':
    main()
