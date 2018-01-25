import operator
import argparse
import numpy as np
import torch

def main(args):
    """
    Compute/print per class accuracy
    """
    itos = torch.load('../classifier/LABELS_vocab_itos.pt')
    perClassAcc_dict = {}
    for eq in itos:
            perClassAcc_dict[str(eq)] == {'right':0, 'wrong':0}

    for k in perClassAcc_dict.keys():
        print(k, perClassAcc.get(k))

def parseArgs():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-truth', type=str, default='../tencent/data/working/basic/val.tsv', help='path to ground true file, usually validation file [default: '']')
    parser.add_argument('-preds', type=str, default='./preds.txt', help='path to preds file [default: '']')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
