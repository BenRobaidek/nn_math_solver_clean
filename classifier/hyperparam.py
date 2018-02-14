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
                        print('classifier cross validation true accuracy:', cross_val_true_acc)

                        ########################################################
                        # seq2seq
                        ########################################################
                        # load s2s predictions
                        s2s_predictions_k1 = np.array([x.strip() == 'True' for x in open('../tencent/data/output/s2s/correctsk1.tsv').readlines()])
                        s2s_predictions_k2 = np.array([x.strip() == 'True' for x in open('../tencent/data/output/s2s/correctsk2.tsv').readlines()])
                        s2s_predictions_k3 = np.array([x.strip() == 'True' for x in open('../tencent/data/output/s2s/correctsk3.tsv').readlines()])
                        s2s_predictions_k4 = np.array([x.strip() == 'True' for x in open('../tencent/data/output/s2s/correctsk4.tsv').readlines()])
                        s2s_predictions_k5 = np.array([x.strip() == 'True' for x in open('../tencent/data/output/s2s/correctsk5.tsv').readlines()])

                        # calculate/print s2s cross validation acc
                        s2s_cross_validation_true_accuracy = np.average([
                            np.sum(s2s_predictions_k1)/len(s2s_predictions_k1),
                            np.sum(s2s_predictions_k2)/len(s2s_predictions_k2),
                            np.sum(s2s_predictions_k3)/len(s2s_predictions_k3),
                            np.sum(s2s_predictions_k4)/len(s2s_predictions_k4),
                            np.sum(s2s_predictions_k5)/len(s2s_predictions_k5)])
                        print('s2s cross validation true accuracy:', s2s_cross_validation_true_accuracy)

                        ########################################################
                        # retrieval
                        ########################################################
                        # load retrieval predictions
                        r_predictions_k1 = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/correctsk1.tsv').readlines()]
                        r_predictions_k2 = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/correctsk2.tsv').readlines()]
                        r_predictions_k3 = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/correctsk3.tsv').readlines()]
                        r_predictions_k4 = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/correctsk4.tsv').readlines()]
                        r_predictions_k5 = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/correctsk5.tsv').readlines()]

                        # calculate/print retrieval cross validation acc
                        retrieval_cross_validation_true_accuracy = np.average([
                            np.sum(r_predictions_k1)/len(r_predictions_k1),
                            np.sum(r_predictions_k2)/len(r_predictions_k2),
                            np.sum(r_predictions_k3)/len(r_predictions_k3),
                            np.sum(r_predictions_k4)/len(r_predictions_k4),
                            np.sum(r_predictions_k5)/len(r_predictions_k5)])
                        print('retrieval cross validation true accuracy:', retrieval_cross_validation_true_accuracy)

                        ########################################################
                        # classifier + seq2seq
                        ########################################################
                        # compute C + S cross val acc
                        class_predictionsk1 = [[x.split()[0] == 'True', float(x.split()[1])] for x in cross_val_results.get(1)[0].get('preds')]
                        class_predictionsk2 = [[x.split()[0] == 'True', float(x.split()[1])] for x in cross_val_results.get(2)[0].get('preds')]
                        class_predictionsk3 = [[x.split()[0] == 'True', float(x.split()[1])] for x in cross_val_results.get(3)[0].get('preds')]
                        class_predictionsk4 = [[x.split()[0] == 'True', float(x.split()[1])] for x in cross_val_results.get(4)[0].get('preds')]
                        class_predictionsk5 = [[x.split()[0] == 'True', float(x.split()[1])] for x in cross_val_results.get(5)[0].get('preds')]

                        c_s2s_predictionsk1 = combineCS(class_predictionsk1, s2s_predictions_k1)
                        c_s2s_predictionsk2 = combineCS(class_predictionsk2, s2s_predictions_k2)
                        c_s2s_predictionsk3 = combineCS(class_predictionsk3, s2s_predictions_k3)
                        c_s2s_predictionsk4 = combineCS(class_predictionsk4, s2s_predictions_k4)
                        c_s2s_predictionsk5 = combineCS(class_predictionsk5, s2s_predictions_k5)

                        c_s2s_cross_validation_true_accuracy = np.average([
                            np.sum(c_s2s_predictionsk1)/len(c_s2s_predictionsk1),
                            np.sum(c_s2s_predictionsk2)/len(c_s2s_predictionsk2),
                            np.sum(c_s2s_predictionsk3)/len(c_s2s_predictionsk3),
                            np.sum(c_s2s_predictionsk4)/len(c_s2s_predictionsk4),
                            np.sum(c_s2s_predictionsk5)/len(c_s2s_predictionsk5)
                            ])
                        print('classifier + s2s cross validation true accuracy:', c_s2s_cross_validation_true_accuracy)

                        ########################################################
                        # retrieval + classifier
                        ########################################################
                        # compute R + C cross val acc
                        # TODO

                        ########################################################
                        # retrieval + seq2seq
                        ########################################################
                        # compute R + S cross val acc
                        # TODO

                        ########################################################
                        # retrieval + classifier + seq2seq
                        ########################################################
                        # compute R + C + S cross val acc
                        # TODO

                    except RuntimeError:
                        print('Oops... Ran out of memory')
                    print('type(cross_val_results)', type(cross_val_results))
                    print('cross_val_results', cross_val_results)
                    #for k in cross_val_results.keys():
                    #    cross_val_results.get(k)[:].pop('preds')
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

def combineCS(class_predictions, s2s_predictions):
    """
    combines classifier and s2s results
    """
    results = dict()
    for thresh in np.multiply(list(range(0,100)), .01):
        results[thresh] = np.sum([c[0] if c[1] > thresh else s for c,s in zip(class_predictions,s2s_predictions)])
    best_thresh = max(results, key=results.get)
    return [c[0] if c[1] > best_thresh else s for c,s in zip(class_predictions,s2s_predictions)]

if __name__ == '__main__':
    main()
