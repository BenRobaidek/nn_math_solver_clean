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
            model.zero_grad()
            inp = batch.text.t()

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

        results = np.append(results, {'avg_loss':avg_loss, 'accuracy':accuracy,
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

def evaluate(data_iter, model, TEXT, emb_dim, LABELS, snis, pred_filter=True):
    model.eval()
    corrects, avg_loss, t5_corrects, rr = 0, 0, 0, 0
    for batch_count,batch in enumerate(data_iter):
        inp, target = batch.text, batch.label
        inp.data.t_()

        logit = model(inp)

        # Filter predictions based on SNI
        if pred_filter:
            mask = np.array(snis * batch.batch_size).reshape(batch.batch_size,-1)
            correct_number_sni = np.array([snis[i] for i in target.data]).transpose()
            for i,column in enumerate(mask.T):
                mask[:,i] = np.equal(correct_number_sni,column)
            mask = torch.LongTensor(mask)
            if torch.cuda.is_available() == 1:
                mask = mask.cuda()
            logit.data[mask == 0] = -sys.maxsize - 1

        loss = F.cross_entropy(logit, target)

        avg_loss += loss.data[0]
        _, preds = torch.max(logit, 1)
        corrects += preds.data.eq(target.data).sum()

        # Rank 5
        _, t5_indices = torch.topk(logit, 5)
        x = torch.unsqueeze(target.data, 1)
        target_index = torch.cat((x, x, x, x, x), 1)
        t5_corrects += t5_indices.data.eq(target_index).sum()
        _, t1_indices = torch.topk(logit, 1)

        _, rank = torch.sort(logit, descending=True)
        target_index = rank.data.eq(torch.unsqueeze(target.data, 1).expand(rank.size()))
        y = torch.arange(1, rank.size()[1]+1).view(1,-1).expand(rank.size())
        cuda = int(torch.cuda.is_available())-1
        if cuda == 0:
            y = y.cuda()
        y = (y.long() * target_index.long()).sum(1).float().reciprocal()
        rr += y.sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = 100.0 * corrects/size
    t5_acc = 100.0 * t5_corrects/size
    mrr = rr/size
    model.train()
    return(avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr)
