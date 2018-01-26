import operator
import argparse
import numpy as np
import torch

def main(args):
    """
    prints combined classifier and seq2seq true accuracy
    """
    s2s_preds = open(args.s2s_preds).readlines()
    s2s_preds = [line.strip() == 'True' for line in s2s_preds]

    classifier_preds = np.array(open(args.classifier_preds).readlines())
    classifier_preds = np.array([line.strip().split(' ') for line in classifier_preds])
    getCombinedAcc(classifier_preds, s2s_preds, 0.1)

def getCombinedAcc(classifier_preds, s2s_preds, threshold):
    print('threshold:', threshold)
    corrects = []
    for classifier_pred, s2s_pred in zip(classifier_preds, s2s_preds):
        print('classifier_pred[0]:', classifier_pred[0])
        print('s2s_preds:', s2s_pred)
        print('classifier_pred[1]:', classifier_pred[1])
        if float(classifier_pred[1]) > threshold:
            corrects = np.append(corrects, [bool(classifier_pred[0])])
        else:
            corrects = np.append(corrects, [bool(s2s_pred)])
    print(classifier_preds[:,0])
    print(np.sum(s2s_preds)/len(corrects))
    print(np.sum(corrects.astype(int))/len(corrects))

def parseArgs():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-s2s_preds', type=str, default='/tmp/predictions.txt.correct', help='path to ground s2s predictions [default: '']')
    parser.add_argument('-classifier_preds', type=str, default='../classifier/predictions.txt', help='path to classifier predictions file [default: '']')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
