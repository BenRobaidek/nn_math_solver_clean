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

    embfix = False # Delete this line later
    pretrained_emb = False # Delete this line later

    cuda = int(torch.cuda.is_available())-1

    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABELS = data.Field(sequential=False)
    VAR_VALUES_VAl = data.Field(sequential=False)
    VAR_VALUES_TEST = data.Field(sequential=False)
    ANS = data.Field(sequential=False)

    train, val, test = data.TabularDataset.splits(
        path=data_path, train=train_path,
        validation=val_path, test=test_path, format='tsv',
        fields=[('text', TEXT), ('label', LABELS), ('var_values', VAR_VALUES_VAL), ('ans', ANS)])

    prevecs = None
    if (pretrained_emb == True):
        print('USING PRETRAINED EMB')
        TEXT.build_vocab(train,vectors=GloVe(name='6B', dim=emb_dim),
                                                                min_freq=mf)
        prevecs=TEXT.vocab.vectors
    else:
        TEXT.build_vocab(train)
    LABELS.build_vocab(train)
    VAR_VALUES_VAL.build_vocab(val)
    VAR_VALUES_TEST.build_vocab(test)
    ANS.build_vocab(val)

    if not os.path.isdir(save_path): os.makedirs(save_path)
    torch.save(LABELS.vocab.itos, save_path + 'LABELS_vocab_itos.pt')

    snis = [eq.count('[') for eq in LABELS.vocab.itos]

    """
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_sizes=(bs, bs, bs),
        sort_key=lambda x: len(x.text))
    """

    train_iter = data.BucketIterator(train, batch_size=bs, sort_key=lambda x: len(x.text), train=True)
    val_iter = data.Iterator(val, batch_size=bs, repeat=False, train=False, sort=False, shuffle=False)
    test_iter = data.Iterator(test, batch_size=bs, repeat=False, train=False, sort=False, shuffle=False)

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
    #print('pretrained_emb:', pretrained_emb)
    #print('embfix:', embfix)
    results = []

    best_true_acc = 0

    for epoch in range(epochs):

        tot_loss = 0
        train_iter.repeat=False
        for batch_count,batch in enumerate(train_iter):
            model.zero_grad()
            inp = batch.text.t()

            preds = model(inp)
            #print(F.softmax(preds))
            loss = criterion(preds, batch.label)
            loss.backward()
            optimizer.step()
            tot_loss += loss.data[0]

        (avg_loss, accuracy, true_acc, corrects, size, t5_acc, t5_corrects, mrr, eval_preds) = evaluate(val_iter, model, TEXT, emb_dim, LABELS, VAR_VALUES_VAL, ANS, snis, pred_filter=pred_filter)
        (_, _, test_true_acc, _, _, _, _, _, test_eval_preds) = evaluate(test_iter, model, TEXT, emb_dim, LABELS, VAR_VALUES_TEST, ANS, snis, pred_filter=pred_filter)

        # save best preds file
        if true_acc > best_true_acc:
            if not os.path.isdir(save_path): os.makedirs(save_path)
            predictions_file = open(save_path + 'predictions.txt', 'w')
            for line in eval_preds:
                predictions_file.write(line + '\n')
            predictions_file.close()

        if save:
            if not os.path.isdir(save_path): os.makedirs(save_path)
            torch.save(model, save_path + '{}_e{}.pt'.format(accuracy, epoch))

        results = np.append(results, {'epoch':epoch, 'avg_loss':avg_loss,
            'accuracy':accuracy, 'true_acc':true_acc, 'corrects':corrects,
            'size': size, 't5_acc':t5_acc, 't5_corrects':t5_corrects, 'mrr':mrr,
            'preds': eval_preds, 'test_eval_preds':test_eval_preds,
            'test_true_acc':test_true_acc})
        if verbose: print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) ' \
                    'true_acc: {:.4f}%(todo/todo) t5_acc: {:.4f}%({}/{}) MRR:' \
                    '{:.6f}\n'.format(avg_loss, accuracy, corrects, size,
                            t5_acc, t5_corrects, size, mrr))

    #print('Best Accuracy:', np.sort([i['accuracy'] for i in results])[-1])
    print('Best True Accuracy:', np.sort([i['true_acc'] for i in results])[-1])
    return results
