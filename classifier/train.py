import os
import argparse
import torch
from torch import autograd, nn
import torch.nn.functional as F
import numpy as np
from numpy import genfromtxt
from torch.autograd import Variable

import model as m
from torchtext import data, datasets
from evalTest import eval,test
from torchtext.vocab import GloVe

def main():
    train(data_path='../tencent/data/working/basic/', train_path='train.tsv',
            val_path='val.tsv', test_path='test.tsv', mf=1, lr=.001, epochs=100,
            bs=8, opt='adam', net_type='lstm', ly=1, hs=100, num_dir=1,
            emb_dim=100, embfix=False, pretrained_emb=False, dropout=0.0,
            pred_filter=True, save_path='.', save=False, folder='',
            acc_thresh=.4, device=0, verbose=False)

def train(data_path, train_path, val_path, test_path, mf, lr, epochs, bs, opt,
            net_type, ly, hs, num_dir, emb_dim, embfix, pretrained_emb,
            dropout, pred_filter, save_path, save, folder, acc_thresh, device,
            verbose=False):
    ###############################################################################
    # Load data
    ###############################################################################

    cuda = int(torch.cuda.is_available())-1

    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABELS = data.Field(sequential=False)

    train, val, test = data.TabularDataset.splits(
        path=data_path, train=train_path,
        validation=val_path, test=test_path, format='tsv',
        fields=[('text', TEXT), ('label', LABELS)])

    prevecs = None
    if (pretrained_emb == True):
        TEXT.build_vocab(train,vectors=GloVe(name='6B', dim=emb_dim),min_freq=mf)
        prevecs=TEXT.vocab.vectors
    else:
        TEXT.build_vocab(train)
    LABELS.build_vocab(train)

    snis = [eq.count('[') for eq in LABELS.vocab.itos]

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(bs, bs, bs),
        sort_key=lambda x: len(x.text))#, device=cuda)

    num_classes = len(LABELS.vocab)
    input_size = len(TEXT.vocab)
    ###############################################################################
    # Build the model
    ###############################################################################

    model = m.Model(input_size=input_size,
                    hidden_size=hs,
                    num_classes=num_classes,
                    prevecs=prevecs,
                    num_layers=ly,
                    num_dir=num_dir,
                    batch_size=bs,
                    emb_dim=emb_dim,
                    embfix=embfix,
                    dropout=dropout,
                    net_type=net_type)
    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    if (opt == 'adamax'):
        optimizer = torch.optim.Adamax(model.parameters())#, lr=args.lr)
    elif (opt == 'adam'):
        optimizer = torch.optim.Adam(model.parameters())#, lr=args.lr)
    elif (opt == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.5)#,lr=args.lr,momentum=0.5)
    else:
        #print('Optimizer unknown, defaulting to adamax')
        optimizer = torch.optim.Adamax(model.parameters())

    ###############################################################################
    # Training the Model
    ###############################################################################
    if cuda == 0:
        model = model.cuda()

    results = []
    for epoch in range(epochs):
        losses = []
        tot_loss = 0
        train_iter.repeat=False
        for batch_count,batch in enumerate(train_iter):
            model.zero_grad()
            inp = batch.text.t()

            preds = model(inp)
            loss = criterion(preds, batch.label)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            tot_loss += loss.data[0]

        #results = np.append(results, [])
        (avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr) = eval(val_iter, model, TEXT, emb_dim, LABELS, snis, pred_filter=pred_filter)
        results = np.append(results, dict{['avg_loss':avg_loss, 'accuracy':accuracy, 'corrects':corrects, 'size':size, 't5_acc':t5_acc, 't5_corrects':t5_corrects, 'mrr':mrr]})
        print(results)
    return results

if __name__ == '__main__':
    main()
