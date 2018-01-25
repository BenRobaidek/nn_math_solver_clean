import operator
import argparse
import numpy as np
import torch

def main(args):
    """
    Compute/print per class accuracy
    """
    itos = torch.load('../classifier/LABELS_vocab_itos.pt')
    predictions = open('../classifier/predictions.txt').readlines()
    dictionary = {}
    for eq in itos:
        dictionary[eq] = [0,0]
    for line in predictions:
        eq, prediction, target = line.split('\t')
        if isFloat(prediction): prediction = float(prediction)
        if isFloat(target): target = float(target)
        print('eq:', eq)
        print('type(eq):',type(eq))
        if isFloat(prediction) and isFloat(target) and abs(prediction - target) <= .002:
            print('type(dictionary):', type(dictionary))
            print('dictionary.get(eq)',dictionary.get(eq))
            dictionary[eq] = np.sum(dictionary.get(eq), [1,0])
        else:
            print('type(dictionary):', type(dictionary))
            dictionary[eq] = np.sum(dictionary.get(eq), [0,1])
    for k in dictionary.keys():
        print(k, dictionary.get(k))

def isFloat(f):
    try:
        float(f)
        return True
    except Exception as e:
        False

def parseArgs():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-truth', type=str, default='../tencent/data/working/basic/val.tsv', help='path to ground true file, usually validation file [default: '']')
    parser.add_argument('-preds', type=str, default='./preds.txt', help='path to preds file [default: '']')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
