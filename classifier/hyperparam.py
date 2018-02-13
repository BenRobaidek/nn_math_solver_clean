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
    ############################################################################
    # PARSE ARGS
    ############################################################################
    parser = argparse.ArgumentParser(description='LSTM text classifier')
    parser.add_argument('-config', type=str,
        default='../tests/exp1/config.json',
        help='config file path [default: ../tests/exp1/config.json]')
    parser.add_argument('-hyperparam_results', type=str,
        default='../tests/exp1/hyperparam_results.json',
        help='hyperparameter search results file path' \
            '[default: ../tests/exp1/hyperparam_results.json]')
    args = parser.parse_args()

    ############################################################################
    # LOAD FROM CONFIG FILE
    ############################################################################
    print('Loading parameters from config file:', args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)

    ############################################################################
    # HYPERPARAM SEARCH
    ############################################################################
    x = list(itertools.product(config['net_type'], config['epochs'],
        config['bs'], config['opt'], config['ly'], config['hs'],
        config['num_dir'], config['embdim'], config['embfix'], config['ptemb'],
        config['dropout'], config['mf'], config['pred_filter']))
    if bool(config['rand']): random.shuffle(x)

    hyperparam_results = None
    try:
        hyperparam_results = json.load(open(args.hyperparam_results, 'r'))
    except FileNotFoundError:
        hyperparam_results = dict()

    print('Hyperparameter search progress: {}% ({}/{})'.format(
        len(hyperparam_results.keys()) / len(x) * 100,
        len(hyperparam_results.keys()),
        len(x)))

    if input("Do you wish to start/continue hyperparameter search? (y/n) ") == "y":
        for (net_type, epoch, bs, opt, ly, hs, num_dir, embdim, embfix, ptemb,
                            dropout, mf, pred_filter) in x:
            if not (embfix and not ptemb):
                hyperparams = {'mf':mf, 'epochs':epoch, 'bs':bs, 'opt':opt,
                            'net_type':net_type, 'ly':ly, 'hs':hs,
                            'num_dir':num_dir, 'embdim':embdim, 'embfix':embfix,
                            'pretrained_emb':ptemb, 'dropout':dropout,
                            'pred_filter':pred_filter}
                results = None
                if not hyperparams in list(hyperparam_results.keys()):
                    cross_val_results = dict()
                    try:

                        # load s2s predictions
                        s2s_predictions_k1 = np.array(x.strip() for x in open('../tencent/data/working/s2s/correctsk1.tsv').readlines()).astype(bool)
                        s2s_predictions_k2 = np.array(x.strip() for x in open('../tencent/data/working/s2s/correctsk2.tsv').readlines()).astype(bool)
                        s2s_predictions_k3 = np.array(x.strip() for x in open('../tencent/data/working/s2s/correctsk3.tsv').readlines()).astype(bool)
                        s2s_predictions_k4 = np.array(x.strip() for x in open('../tencent/data/working/s2s/correctsk4.tsv').readlines()).astype(bool)
                        s2s_predictions_k5 = np.array(x.strip() for x in open('../tencent/data/working/s2s/correctsk5.tsv').readlines()).astype(bool)
                        print('s2s cross validation true accuracy:',
                                np.average([np.sum(s2s_predictions_k1),
                                np.sum(s2s_predictions_k2),
                                np.sum(s2s_predictions_k3),
                                np.sum(s2s_predictions_k4),
                                np.sum(s2s_predictions_k5)]))

                        save_path = ''
                        for k in hyperparams.keys():
                            save_path = save_path + str(k) + str(hyperparams[k])
                        for i in range(1,6):
                            results = train(data_path=config['data_path'],
                                    train_path='traink' + str(i) + '.tsv',
                                    val_path='valk' + str(i) + '.tsv',
                                    test_path='test.tsv',
                                    mf=mf, epochs=epoch, bs=bs, opt=opt,
                                    net_type=net_type,
                                    ly=ly, hs=hs, num_dir=num_dir, emb_dim=embdim,
                                    embfix=bool(embfix), pretrained_emb=bool(ptemb),
                                    dropout=dropout, pred_filter=bool(pred_filter),
                                    save_path='./hyperparam_results/' + save_path + '/', save=False, verbose=False)
                            results = sorted(results, key=lambda x: x['true_acc'], reverse=True)
                            cross_val_results[str(i)] = results
                        cross_val_true_acc = np.average([x[0].get('true_acc') for x in list(cross_val_results.values())])
                        cross_val_results['cross_val_true_acc'] = cross_val_true_acc



                        # load retrieval predictions
                        # TODO

                        # compute C + S cross val acc
                        # TODO

                        # compute R + C cross val acc
                        # TODO

                        # compute R + S cross val acc
                        # TODO

                        # compute R + C + S cross val acc
                        # TODO

                    except RuntimeError:
                        print('Oops... Ran out of memory')
                    hyperparam_results[str(hyperparams)] = cross_val_results

            with open(args.hyperparam_results, 'w') as f:
                json.dump(hyperparam_results, f, indent=2)

    ############################################################################
    # RETRAIN/SAVE BEST MODEL
    ############################################################################
    if input('Do you wish to train the best model found thus far? (y/n)? ') == 'y':
        print('hyperparam_results:', hyperparam_results)

        #best_hyperparams = sorted(hyperparams_results, key=lambda x: x.values()['accuracy'],
        #        reverse=True)
        train(data_path=config['data_path'],
                train_path='train.tsv',
                val_path='val.tsv', test_path='test.tsv', mf=mf,
                epochs=epoch, bs=bs, opt=opt, net_type=net_type,
                ly=ly, hs=hs, num_dir=num_dir, emb_dim=embdim,
                embfix=bool(embfix), pretrained_emb=bool(ptemb),
                dropout=dropout, pred_filter=bool(pred_filter),
                save_path='./', save=False, verbose=False)


if __name__ == '__main__':
    main()
