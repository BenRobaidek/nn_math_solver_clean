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

    for eq in itos:
        print(eq)

    results = np.ones([len(itos),2])
    results = -1 * results

    for line in predictions:
        equation, prediction, target = line.split('\t')
        right = results[itos.index(equation),0]
        wrong = results[itos.index(equation),1]
        if right == -1 or wrong == -1:
            right = 0
            wrong = 0
            # make sure this updates asarray
        if isFloat(prediction): prediction = float(prediction)
        if isFloat(target): target = float(target)
        if isFloat(prediction) and isFloat(target) and abs(prediction - target) <= .002:
            right += 1
        else:
            wrong += 1
        results[itos.index(equation),0] = right
        results[itos.index(equation),0] = wrong

    for line in results:
        print(line)
    #print(results)
    print('len(itos):', np.unique(len(itos)))

    """
    for k in dictionary.keys():
        true_acc = dictionary.get(k)[0] / (dictionary.get(k)[0] + dictionary.get(k)[1])
        print(k, 'true acc:', true_acc)
    """

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
