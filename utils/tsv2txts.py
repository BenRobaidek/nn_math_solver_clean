import argparse
import numpy as np
import torch
from torch import autograd, nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
from torchtext import data

sys.path.append('../tencent/models/')
sys.path.append('../classifier/')

def main(args):
    print('Running main...')
    inp = [line.split('\t')[1].strip() for line in open(args.inp).readlines()]
    model = torch.load(args.model, map_location=lambda storage, loc: storage.cuda(0))
    model.eval()

    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABELS = data.Field(sequential=False)

    train = data.TabularDataset(path=args.train, format='tsv', fields=[('text', TEXT), ('label', LABELS)])
    inp = data.TabularDataset(path=args.inp, format='tsv', fields=[('text', TEXT), ('label', LABELS)])

    TEXT.build_vocab(train)
    LABELS.build_vocab(train)

    inp_iter = data.BucketIterator(inp, batch_size=8, sort_key=lambda x: len(x.text), repeat=False)

    predictions = open(args.preds, 'w')
    for batch in inp_iter:
        logit = model(batch.text.t())
        _, preds = torch.max(logit, 1)
        for ID in preds:
            predictions.write(LABELS.vocab.itos[ID.data[0]] + '\n')
    predictions.close()

def parseArgs():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-inp', type=str, default='../tencent/models/good_model.pt', help='path to model [default: \'../tencent/models/good_model.pt\']')
    parser.add_argument('-output-src', type=str, default='../tencent/data/working/basic/train.tsv', help='path to train tsv file [default: \'../tencent/data/working/basic/train.tsv\']')
    parser.add_argument('-output-tgt', type=str, default='../tencent/data/working/basic/val.tsv', help='path to input tsv file, usually validation file [default: \'../tencent/data/working/basic/val.tsv\']')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    print('args:', args)
    main(args)
