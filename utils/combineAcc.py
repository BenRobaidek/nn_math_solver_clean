import operator
import argparse
import numpy as np
import torch

def main(args):
    """
    prints combined classifier and seq2seq true accuracy
    """
    s2s_preds = open(args.s2s_preds).readlines()
    s2s_preds = [bool(line.strip()) for line in s2s_preds]

    classifier_preds = open(args.classifier_preds).readlines()
    classifier_preds = [line.split('\t') for line in classifier_preds]

    print(s2s_preds)
    print(classifier_preds)

def parseArgs():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-s2s_preds', type=str, default='/tmp/predictions.txt.correct', help='path to ground s2s predictions [default: '']')
    parser.add_argument('-classifier_preds', type=str, default='../classifier/predictions.txt', help='path to classifier predictions file [default: '']')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
