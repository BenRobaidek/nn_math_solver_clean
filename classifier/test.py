import os
import argparse
import torch
from torch import autograd, nn
import torch.nn.functional as F
from numpy import genfromtxt
from torch.autograd import Variable

import model as m
from torchtext import data, datasets
from evalTest import eval,test
from torchtext.vocab import GloVe

def main():
    args = parseArgs()

    cuda = int(torch.cuda.is_available())

    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABELS = data.Field(sequential=False)

    train, truth, preds = data.TabularDataset.splits(
        path='', train=args.train, validation=args.truth, test=args.preds, format='tsv',
        fields=[('text', TEXT), ('label', LABELS)])

    TEXT.build_vocab(train)
    LABELS.build_vocab(train)

    train_iter, truth_iter, preds_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(8, 8, 8),
        sort_key=lambda x: len(x.text), repeat=False)

    model = torch.load(args.model)

    (avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr) = eval(truth_iter, model, TEXT, 300, LABELS)
    print('COMMON ACCURACY:', accuracy)
    (avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr) = eval(preds_iter, model, TEXT, 300, LABELS)
    print('UNCOMMON ACCURACY:', accuracy)

    perClassAcc(args.truth, args.preds)

def perClassAcc(truth_path, preds_path):
    """
    Compute/print per class accuracy
    """
    truth = open(truth_path).readlines()
    preds = open(preds_path).readlines()
    print(truth)
    print(preds)


def parseArgs():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('-model', type=str, default='../tencent/models/good_model.pt', help='path to model, [default: \'../tencent/models/good_model.pt\']')
    parser.add_argument('-train', type=str, default='../tencent/data/', help='path to train data file, [default: \'\']')
    parser.add_argument('-truth', type=str, default='', help='path to ground true file, usually validation file [default: '']')
    parser.add_argument('-preds', type=str, default='', help='path to preds file [default: '']')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
