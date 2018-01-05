import operator
import argparse
import numpy as np

def main(args):
    """
    Compute/print per class accuracy
    """
    truth = [line.strip().split('\t')[1] for line in open(args.truth).readlines()]
    preds = [line.strip() for line in open(args.preds).readlines()]

    # initialize dict
    perClassAcc = dict.fromkeys(np.unique(truth))

    # populate dict
    for line in zip(truth,preds):
        if perClassAcc[line[0]] is None:
            perClassAcc[line[0]] = np.array([int(line[0] == line[1]), int(not line[0] == line[1]), 0]).astype(float)
        else:
            perClassAcc[line[0]] = np.add(perClassAcc[line[0]], [int(line[0] == line[1]), int(not line[0] == line[1]), 0])

    # compute per class accuracies
    print(perClassAcc)
    for key in perClassAcc.keys():
        perClassAcc[key][2] = perClassAcc[key][0] / (perClassAcc[key][0] + perClassAcc[key][1])

    # Sort keys
    keylist = sorted(perClassAcc, key=lambda x: perClassAcc.get(x)[2], reverse=True)

    #
    for key in keylist:
        print('{:>60}   {:>5}  ({}/{})'.format(key.strip(), perClassAcc[key][2], int(perClassAcc[key][0]), int(perClassAcc[key][0] + perClassAcc[key][1])))

def parseArgs():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('-truth', type=str, default='../tencent/data/working/basic/val.tsv', help='path to ground true file, usually validation file [default: '']')
    parser.add_argument('-preds', type=str, default='./preds.txt', help='path to preds file [default: '']')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
