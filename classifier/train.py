import os
import argparse
import inspect
import torch
from torch import autograd, nn
import torch.nn.functional as F
import numpy as np
from numpy import genfromtxt
from torch.autograd import Variable

import model as m
from evaluate import evaluate
from torchtext import data, datasets
from torchtext.vocab import GloVe

def train(data_path, train_path, val_path, test_path, mf, epochs, bs, opt,
            net_type, ly, hs, num_dir, emb_dim, embfix, pretrained_emb,
            dropout, pred_filter, save_path, save, verbose=False):
    ############################################################################
    # Load data
    ############################################################################

    cuda = int(torch.cuda.is_available())-1

    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABELS = data.Field(sequential=False)

    train, val, test = data.TabularDataset.splits(
        path=data_path, train=train_path,
        validation=val_path, test=test_path, format='tsv',
        fields=[('text', TEXT), ('label', LABELS)])

    prevecs = None
    if (pretrained_emb == True):
        TEXT.build_vocab(train,vectors=GloVe(name='6B', dim=emb_dim),
                                                                min_freq=mf)
        prevecs=TEXT.vocab.vectors
    else:
        TEXT.build_vocab(train)
    LABELS.build_vocab(train)

    snis = [eq.count('[') for eq in LABELS.vocab.itos]

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(bs, bs, bs),
        sort_key=lambda x: len(x.text))

    num_classes = len(LABELS.vocab)
    input_size = len(TEXT.vocab)
    ############################################################################
    # Build the model
    ############################################################################

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
        optimizer = torch.optim.Adamax(model.parameters())
    elif (opt == 'adam'):
        optimizer = torch.optim.Adam(model.parameters())
    elif (opt == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.5)
    else:
        #print('Optimizer unknown, defaulting to adamax')
        optimizer = torch.optim.Adamax(model.parameters())

    ############################################################################
    # Training the Model
    ############################################################################
    if cuda == 0:
        model = model.cuda()

    hyperparams = {'mf':mf, 'epochs':epochs, 'bs':bs, 'opt':opt,
                'net_type':net_type, 'ly':ly, 'hs':hs, 'num_dir':num_dir,
                'emb_dim':emb_dim, 'embfix':embfix,
                'pretrained_emb':pretrained_emb, 'dropout':dropout,
                'pred_filter':pred_filter}
    print('Training:', hyperparams)
    results = []
    for epoch in range(epochs):
        losses = []
        tot_loss = 0
        train_iter.repeat=False
        for batch_count,batch in enumerate(train_iter):
            print('batch_count:', batch_count)
            model.zero_grad()
            inp = batch.text.t()

            # Check with Rik about this
            #inp.requires_grad=True

            preds = model(inp)
            loss = criterion(preds, batch.label)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            tot_loss += loss.data[0]

        (avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr) = evaluate(val_iter, model, TEXT, emb_dim, LABELS, snis, pred_filter=pred_filter)

        if save:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(model, save_path + '{}_e{}.pt'.format(accuracy, epoch))

        results = np.append(results, {'epoch':epoch, 'avg_loss':avg_loss, 'accuracy':accuracy,
            'corrects':corrects, 'size': size, 't5_acc':t5_acc,
            't5_corrects':t5_corrects, 'mrr':mrr})
        if verbose: print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) ' \
                    't5_acc: {:.4f}%({}/{}) MRR: {:.6f}\n'.format(avg_loss,
                                                                accuracy,
                                                                corrects,
                                                                size,
                                                                t5_acc,
                                                                t5_corrects,
                                                                size,
                                                                mrr))
    print('Accuracy:', np.sort([i['accuracy'] for i in results])[-1])
    return results
