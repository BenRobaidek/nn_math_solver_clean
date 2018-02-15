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
                        for i in range(0,5):
                            train_path = 'traink' + str(i%5+1) + str((i+1)%5+1) + str((i+2)%5+1) + '.tsv'
                            val_path = 'valk' + str((i+3)%5+1) + '.tsv'
                            test_path = 'testk' + str((i+4)%5+1) + '.tsv'
                            print('train:', train_path, 'val:', val_path, 'test:', test_path)
                            results = train(data_path=config['data_path'],
                                    train_path=train_path,
                                    val_path=val_path,
                                    test_path=test_path,
                                    mf=mf, epochs=epoch, bs=bs, opt=opt,
                                    net_type=net_type,
                                    ly=ly, hs=hs, num_dir=num_dir, emb_dim=embdim,
                                    embfix=bool(embfix), pretrained_emb=bool(ptemb),
                                    dropout=dropout, pred_filter=bool(pred_filter),
                                    save_path='./hyperparam_results/' + save_path + '/', save=False, verbose=False)
                            results = sorted(results, key=lambda x: x['true_acc'], reverse=True)
                            cross_val_results[i] = results
                        val_cross_val_true_acc = np.average([x[0].get('true_acc') for x in list(cross_val_results.values())])
                        test_cross_val_true_acc = np.average([x[0].get('test_true_acc') for x in list(cross_val_results.values())])
                        cross_val_results['val_cross_val_true_acc'] = val_cross_val_true_acc
                        cross_val_results['test_cross_val_true_acc'] = test_cross_val_true_acc
                        print('classifier cross validation true accuracy (VAL):', val_cross_val_true_acc)
                        print('classifier cross validation true accuracy (TEST):', test_cross_val_true_acc)

                        ########################################################
                        # seq2seq
                        ########################################################
                        # load s2s predictions
                        s2s_predictions_valk1 = np.array([x.strip() == 'True' for x in open('../tencent/data/output/s2s/correctsk1.tsv').readlines()])
                        s2s_predictions_valk2 = np.array([x.strip() == 'True' for x in open('../tencent/data/output/s2s/correctsk2.tsv').readlines()])
                        s2s_predictions_valk3 = np.array([x.strip() == 'True' for x in open('../tencent/data/output/s2s/correctsk3.tsv').readlines()])
                        s2s_predictions_valk4 = np.array([x.strip() == 'True' for x in open('../tencent/data/output/s2s/correctsk4.tsv').readlines()])
                        s2s_predictions_valk5 = np.array([x.strip() == 'True' for x in open('../tencent/data/output/s2s/correctsk5.tsv').readlines()])

                        # calculate/print s2s cross validation acc
                        s2s_cross_validation_true_accuracy_val = np.average([
                            np.sum(s2s_predictions_valk1)/len(s2s_predictions_valk1),
                            np.sum(s2s_predictions_valk2)/len(s2s_predictions_valk2),
                            np.sum(s2s_predictions_valk3)/len(s2s_predictions_valk3),
                            np.sum(s2s_predictions_valk4)/len(s2s_predictions_valk4),
                            np.sum(s2s_predictions_valk5)/len(s2s_predictions_valk5)])
                        cross_val_results['s2s_cross_validation_true_accuracy_val'] = s2s_cross_validation_true_accuracy_val
                        cross_val_results['s2s_cross_validation_true_accuracy_test'] = None #TODO
                        print('s2s cross validation true accuracy (VAL):', s2s_cross_validation_true_accuracy_val)
                        print('s2s cross validation true accuracy (TEST):', 'TODO')

                        ########################################################
                        # retrieval
                        ########################################################
                        # load retrieval predictions
                        r_predictions_valk1 = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/correctsk1.tsv').readlines()]
                        r_predictions_valk2 = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/correctsk2.tsv').readlines()]
                        r_predictions_valk3 = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/correctsk3.tsv').readlines()]
                        r_predictions_valk4 = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/correctsk4.tsv').readlines()]
                        r_predictions_valk5 = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/correctsk5.tsv').readlines()]

                        # calculate/print retrieval cross validation acc
                        retrieval_cross_validation_true_accuracy_val = np.average([
                            np.sum(r_predictions_valk1)/len(r_predictions_valk1),
                            np.sum(r_predictions_valk2)/len(r_predictions_valk2),
                            np.sum(r_predictions_valk3)/len(r_predictions_valk3),
                            np.sum(r_predictions_valk4)/len(r_predictions_valk4),
                            np.sum(r_predictions_valk5)/len(r_predictions_valk5)])
                        cross_val_results['retrieval_cross_validation_true_accuracy_val'] = retrieval_cross_validation_true_accuracy_val
                        cross_val_results['retrieval_cross_validation_true_accuracy_test'] = None #TODO
                        print('retrieval cross validation true accuracy (VAL):', retrieval_cross_validation_true_accuracy_val)
                        print('retrieval cross validation true accuracy (TEST):', 'TODO')

                        ########################################################
                        # classifier + seq2seq
                        ########################################################
                        # compute C + S cross val acc
                        class_predictions_valk1 = [[x.split()[0] == 'True', float(x.split()[1])] for x in cross_val_results.get(0)[0].get('preds')]
                        class_predictions_valk2 = [[x.split()[0] == 'True', float(x.split()[1])] for x in cross_val_results.get(1)[0].get('preds')]
                        class_predictions_valk3 = [[x.split()[0] == 'True', float(x.split()[1])] for x in cross_val_results.get(2)[0].get('preds')]
                        class_predictions_valk4 = [[x.split()[0] == 'True', float(x.split()[1])] for x in cross_val_results.get(3)[0].get('preds')]
                        class_predictions_valk5 = [[x.split()[0] == 'True', float(x.split()[1])] for x in cross_val_results.get(4)[0].get('preds')]

                        best_thresh = getThresh(
                                class_predictions_valk1 +
                                class_predictions_valk2 +
                                class_predictions_valk3 +
                                class_predictions_valk4 +
                                class_predictions_valk5),
                                s2s_predictions_valk1 +
                                s2s_predictions_valk2 +
                                s2s_predictions_valk3 +
                                s2s_predictions_valk4 +
                                s2s_predictions_valk5
                                ))
                        print('best_thresh:', best_thresh)
                        print('classifier + s2s cross validation true accuracy (VAL):', c_s2s_cross_validation_true_accuracy)

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
                    for k in cross_val_results.keys():
                        try:
                            for x in cross_val_results[k]:
                                x.pop('preds')
                                x.pop('test_preds')
                        except TypeError:
                            pass
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

def getThresh(class_predictions, s2s_predictions):
    """
    combines classifier and s2s results
    """
    results = dict()
    for thresh in np.multiply(list(range(0,100)), .01):
        results[thresh] = np.sum([c[0] if c[1] > thresh else s for c,s in zip(class_predictions,s2s_predictions)])
    return max(results, key=results.get)

def combineCS(class_predictions, s2s_predictions, thresh):
    return [c[0] if c[1] > best_thresh else s for c,s in zip(class_predictions,s2s_predictions)]

if __name__ == '__main__':
    main()
