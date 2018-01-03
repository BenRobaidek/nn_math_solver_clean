import argparse
import numpy as np

def main(args):
    """
    Compute/print per class accuracy
    """
    truth = open(args.truth).readlines()
    preds = open(args.preds).readlines()

    # initialize dict
    perClassAcc = dict.fromkeys(np.unique(truth))

    # populate dict
    for line in zip(truth,preds):
        if perClassAcc[line[0]] is None:
            perClassAcc[line[0]] = [int(line[0] == line[1]), int(not line[0] == line[1])]
        else:
            perClassAcc[line[0]] = np.add(perClassAcc[line[0]], [int(line[0] == line[1]), int(not line[0] == line[1])])

    print(perClassAcc)

    # Sort keys
    left off here

    #
    for key in perClassAcc:
        acc = str(perClassAcc[key][0] / (perClassAcc[key][0] + perClassAcc[key][1]))
        print('{:>12}   {:>12}  ({}/{})'.format(key.strip(), acc, perClassAcc[key][0], perClassAcc[key][0] + perClassAcc[key][1]))

def parseArgs():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('-truth', type=str, default='./truth.txt', help='path to ground true file, usually validation file [default: '']')
    parser.add_argument('-preds', type=str, default='./preds.txt', help='path to preds file [default: '']')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
