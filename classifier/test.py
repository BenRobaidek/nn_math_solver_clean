import os
import argparse
import torch
from torch import autograd, nn
import torch.nn.functional as F
from numpy import genfromtxt
from torch.autograd import Variable

import data
import model as m
from torchtext import data, datasets
import mydatasets
from evalTest import eval,test
from torchtext.vocab import GloVe
from vecHandler import Vecs

def main():
    cuda = int(torch.cuda.is_available())-1

    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABELS = data.Field(sequential=False)

    train, val, test = data.TabularDataset.splits(
        path='../tencent/data/', train='train_0.8.tsv',
        validation='train_dev_0.8_common.tsv', test='train_dev_0.8_uncommon.tsv', format='tsv',
        fields=[('text', TEXT), ('label', LABELS)])

    TEXT.build_vocab(train)
    LABELS.build_vocab(train)

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(8, 8, 8),
        sort_key=lambda x: len(x.text), repeat=False)

    model = torch.load('../tencent/models/common_0.8/net-lstm_e25_bs8_opt-adam_ly1_hs300_dr2_ed200_fembFalse_ptembFalse_drp0.3/acc47.60_e6.pt')

    (avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr) = eval(val_iter, model, TEXT, 300)
    print('COMMON ACCURACY:', accuracy)
    (avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr) = eval(test_iter, model, TEXT, 300)
    print('UNCOMMON ACCURACY:', accuracy)

if __name__ == '__main__':
    main()
