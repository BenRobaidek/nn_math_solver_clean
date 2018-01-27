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
    classifier_preds = np.array([line.split(' ') for line in classifier_preds])
    classifier_probabilities = np.array(classifier_preds[:,1])
    classifier_probabilities = np.array([line.strip() for line in classifier_probabilities]).astype(float)
    classifier_preds = [line == 'True' for line in classifier_preds[:,0]]
    print('classifier_probabilities:', classifier_probabilities)
    print('classifier_preds:', classifier_preds)
    print('s2s_preds:', s2s_preds)
    getCombinedAcc(classifier_probabilities, classifier_preds, s2s_preds, 0.1)

def getCombinedAcc(classifier_probabilities, classifier_preds, s2s_preds, threshold):
    print('threshold:', threshold)
    corrects = []
    for probability, classifier_pred, s2s_pred in zip(probability, classifier_pred, s2s_pred):
        print(probability)
        print(classifier_pred)
        print(s2s_pred)

def parseArgs():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-s2s_preds', type=str, default='/tmp/predictions.txt.correct', help='path to ground s2s predictions [default: '']')
    parser.add_argument('-classifier_preds', type=str, default='../classifier/predictions.txt', help='path to classifier predictions file [default: '']')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
