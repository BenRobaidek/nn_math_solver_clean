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
        default='../tests/exp1/hyperparam_results_kushman.json',
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

                        train_path = 'traink1234.tsv'
                        val_path = 'valk1234.tsv'
                        test_path = 'testk5.tsv'
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
                        #results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

                        ########################################################
                        # classifier
                        ########################################################
                        # load classifier predictions
                        classifier_validation_predictions = [[x.split()[0] == 'True', float(x.split()[1])] for x in results[0].get('preds')]
                        classifier_test_predictions = [[x.split()[0] == 'True', float(x.split()[1])] for x in results[0].get('test_eval_preds')]

                        results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
                        print('Classification acc (TEST):', results[0].get('test_acc'))
                        print('classifier true acc (VAL):', 100 * (np.sum(np.array(classifier_validation_predictions)[:,0])/len(np.array(classifier_validation_predictions)[:,0])))
                        print('classifier true acc (TEST):', 100 * (np.sum(np.array(classifier_test_predictions)[:,0])/len(np.array(classifier_test_predictions)[:,0])))

                        ########################################################
                        # seq2seq
                        ########################################################
                        # load s2s predictions
                        print('config[data_path]:', config['data_path'])
                        s2s_validation_predictions_path = config['data_path'].strip('/working/basic/') + 'a/output/s2s_basic/corrects_valk1234.txt'
                        s2s_test_predictions_path = config['data_path'].strip('/working/basic/') + 'a/output/s2s_basic/corrects_testk5.txt'
                        s2s_validation_predictions = np.array([x.strip() == 'True' for x in open(s2s_validation_predictions_path).readlines()])
                        s2s_test_predictions = np.array([x.strip() == 'True' for x in open(s2s_test_predictions_path).readlines()])
                        print('s2s true acc (VAL):', 100 * (np.sum(s2s_validation_predictions.astype(int))/len(s2s_validation_predictions)))
                        print('s2s true acc (TEST):', 100 * (np.sum(s2s_test_predictions.astype(int))/len(s2s_test_predictions)))

                        ########################################################
                        # retrieval
                        ########################################################
                        # load retrieval predictions
                        retrieval_validation_predictions = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/val.correct.txt').readlines()]
                        retrieval_test_predictions = [[x.strip().split()[0] == 'True', float(x.strip().split()[1])] for x in open('../tencent/data/output/retrieval/test.correct.txt').readlines()]
                        print('r true accuracy (VAL):', np.sum(np.array(retrieval_validation_predictions)[:,0])/len(np.array(retrieval_validation_predictions)[:,0]))
                        print('r true accuracy (TEST):', np.sum(np.array(retrieval_test_predictions)[:,0])/len(np.array(retrieval_test_predictions)[:,0]))
                        #print(retrieval_validation_predictions)
                        #print(retrieval_test_predictions)
                        #print('retrieval validation acc:', '')

                        ########################################################
                        # classifier + seq2seq
                        ########################################################
                        # compute C + S cross val acc

                        best_thresh = getThresh(
                                classifier_validation_predictions,
                                s2s_validation_predictions
                                )
                        print('best_thresh:', best_thresh)

                        classifier_s2s_validation_predictions = combineCS(
                                classifier_validation_predictions,
                                s2s_validation_predictions,
                                thresh=best_thresh
                                )

                        classifier_s2s_test_predictions = combineCS(
                                classifier_test_predictions,
                                s2s_test_predictions,
                                thresh=best_thresh
                                )

                        print('classifier + s2s true accuracy (VAL):', np.sum(classifier_s2s_validation_predictions)/len(classifier_s2s_validation_predictions))
                        print('classifier + s2s true accuracy (TEST):', np.sum(classifier_s2s_test_predictions)/len(classifier_s2s_test_predictions))

                        # print questions that both classifier and s2s get wrong
                        val_problems = open(config['data_path'] + val_path).readlines()
                        #print(classifier_s2s_validation_predictions)

                        #for i,j in zip(classifier_s2s_validation_predictions,val_problems):
                        #    if i: print(j)


                        ########################################################
                        # retrieval + classifier
                        ########################################################
                        # compute R + C cross val acc
                        best_thresh = getThresh(retrieval_validation_predictions, np.array(classifier_validation_predictions)[:,0])
                        print('best_thresh:', best_thresh)
                        classifier_r_validation_predictions = combineCS(
                                retrieval_validation_predictions,
                                np.array(classifier_validation_predictions)[:,0],
                                thresh=best_thresh
                                )
                        classifier_r_test_predictions = combineCS(
                                retrieval_test_predictions,
                                np.array(classifier_test_predictions)[:,0],
                                thresh=best_thresh
                                )
                        #print(classifier_r_validation_predictions)
                        #print(classifier_r_test_predictions)
                        classifier_r_validation_predictions = np.array(classifier_r_validation_predictions)#[:,0]
                        #print(classifier_r_validation_predictions)
                        print('r + classifier true accuracy (VAL):', np.sum(classifier_r_validation_predictions)/len(classifier_r_validation_predictions))
                        print('r + classifier true accuracy (TEST):', np.sum(classifier_r_test_predictions)/len(classifier_r_test_predictions))

                        ########################################################
                        # classifier + retrieval
                        ########################################################
                        # compute R + C cross val acc
                        best_thresh = getThresh(classifier_validation_predictions, np.array(retrieval_validation_predictions)[:,0])
                        print('best_thresh:', best_thresh)
                        classifier_r_validation_predictions = combineCS(
                                classifier_validation_predictions,
                                np.array(retrieval_validation_predictions)[:,0],
                                thresh=best_thresh
                                )
                        classifier_r_test_predictions = combineCS(
                                classifier_test_predictions,
                                np.array(retrieval_test_predictions)[:,0],
                                thresh=best_thresh
                                )
                        #print(classifier_r_validation_predictions)
                        #print(classifier_r_test_predictions)
                        classifier_r_validation_predictions = np.array(classifier_r_validation_predictions)#[:,0]
                        #print(classifier_r_validation_predictions)
                        print('classifier + r true accuracy (VAL):', np.sum(classifier_r_validation_predictions)/len(classifier_r_validation_predictions))
                        print('classifier + r true accuracy (TEST):', np.sum(classifier_r_test_predictions)/len(classifier_r_test_predictions))


                        ########################################################
                        # retrieval + seq2seq
                        ########################################################
                        # compute R + S cross val acc
                        # TODO

                        ########################################################
                        # retrieval + classifier + seq2seq
                        ########################################################
                        # compute R + C + S cross val acc


                    except RuntimeError:
                        print('Oops... Ran out of memory')
                    for k in cross_val_results.keys():
                        try:
                            for x in cross_val_results[k]:
                                x.pop('preds')
                                x.pop('test_eval_preds')
                        except TypeError:
                            pass
                    hyperparam_results[str(hyperparams)] = cross_val_results

            with open(args.hyperparam_results, 'w') as f:
                json.dump(hyperparam_results, f, indent=2)

    ############################################################################
    # RETRAIN/SAVE BEST MODEL
    ############################################################################
    if input('Do you wish to report the best model found thus far? (y/n)? ') == 'y':
        print('hyperparam_results:', hyperparam_results)

        #best_hyperparams = sorted(hyperparams_results, key=lambda x: x.values()['accuracy'],
        #        reverse=True)

        """
        train(data_path=config['data_path'],
                train_path='train.tsv',
                val_path='val.tsv', test_path='test.tsv', mf=mf,
                epochs=epoch, bs=bs, opt=opt, net_type=net_type,
                ly=ly, hs=hs, num_dir=num_dir, emb_dim=embdim,
                embfix=bool(embfix), pretrained_emb=bool(ptemb),
                dropout=dropout, pred_filter=bool(pred_filter),
                save_path='./', save=False, verbose=False)
        """

def getThresh(class_predictions, s2s_predictions):
    """
    combines classifier and s2s results
    """
    print(len(class_predictions), len(s2s_predictions))
    assert len(class_predictions) == len(s2s_predictions)
    results = dict()
    for thresh in np.multiply(list(range(0,100)), .01):
        results[thresh] = np.sum([c[0] if c[1] > thresh else s for c,s in zip(class_predictions,s2s_predictions)])
        #print(np.sum([c[0] if c[1] > thresh else s for c,s in zip(class_predictions,s2s_predictions)]))
    return max(results, key=results.get)

def getThreshCSR(class_predictions, s2s_predictions, r_predictions):
    results = dict()
    assert len(class_predictions) == len(s2s_predictions)
    assert len(class_predictions) == len(r_predictions)
    for class_thresh in np.multiply(list(range(0,100)), .01):
        for r_thresh in np.multiply(list(range(0,100)), .01):
            results[[class_thresh,r_thresh]] = np.sum([r[0] if r[1] > r_thresh else c[0] if c[1] > thresh else s for c,s,r in zip(class_predictions,s2s_predictions,r_predictions)])
    return max(results, key=results.get)

def combineCSR(class_predictions, s2s_predictions, r_predictions, r_thresh, c_thresh):
    assert len(class_predictions) == len(s2s_predictions)
    return [r[0] if r[1] > r_thresh else c[0] if c[1] > c_thresh else s for c,s,r in zip(class_predictions,s2s_predictions,r_predictions)]

def combineCS(class_predictions, s2s_predictions, thresh):
    assert len(class_predictions) == len(s2s_predictions)
    return [c[0] if c[1] > thresh else s for c,s in zip(class_predictions,s2s_predictions)]

if __name__ == '__main__':
    main()
