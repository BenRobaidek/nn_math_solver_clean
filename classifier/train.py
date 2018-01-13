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
    """
    args = parseParams()
    if not os.path.isdir(args.save_path_full):
        train(args)
    else:
        print('Previously Trained')
    """
    train(data_path='../tencent/data/working/basic/', train_path='train.tsv',
            val_path='val.tsv', test_path='test.tsv', mf=1, lr=.001, epochs=100,
            bs=8, opt='adam', net_type='lstm', ly=2, hs=300, num_dir=2,
            emb_dim=300, embfix=False, pretrained_emb=False, dropout=0.0,
            pred_filter=True, save_path='./', save=False, folder='',
            acc_thresh=.4, device=0, verbose=False)

def train(data_path, train_path, val_path, test_path, mf, lr, epochs, bs, opt,
            net_type, ly, hs, num_dir, emb_dim, embfix, pretrained_emb,
            dropout, pred_filter, save_path, save, folder, acc_thresh, device,
            verbose=False):
    ###############################################################################
    # Load data
    ###############################################################################
    print('MADE IT HERE')
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

    highest_t1_acc = 0
    highest_t1_acc_metrics = ''
    highest_t1_acc_params = ''
    results = ''
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

            #if (batch_count % 20 == 0):
            #    print('Batch: ', batch_count, '\tLoss: ', str(losses[-1].data[0]))
        #print('Average loss over epoch ' + str(epoch) + ': ' + str(tot_loss/len(losses)))
        (avg_loss, accuracy, corrects, size, t5_acc, t5_corrects, mrr) = eval(val_iter, model, TEXT, emb_dim, LABELS, snis, pred_filter=pred_filter)
        if accuracy > acc_thresh:
            save_path = '{}/acc{:.2f}_e{}.pt'.format(save_path, accuracy, epoch)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(model, save_path)

        if highest_t1_acc < accuracy:
            highest_t1_acc = accuracy
            highest_t1_acc_metrics = ('acc: {:6.4f}%({:3d}/{}) EPOCH{:2d} - loss: {:.4f} t5_acc: {:6.4f}%({:3d}' \
                    '/{}) MRR: {:.6f}'.format(accuracy, corrects, size,epoch, avg_loss, t5_acc, t5_corrects, size, mrr))

            highest_t1_acc_params = (('PARAMETERS:' \
                    'net-%s' \
                    '_e%i' \
                    '_bs%i' \
                    '_opt-%s' \
                    '_ly%i' \
                    '_hs%i' \
                    '_dr%i'
                    '_ed%i' \
                    '_femb%s' \
                    '_ptemb%s' \
                    '_drp%.1f' \
                    '_mf%d\n'
                    % (net_type, epochs, batch_size, opt, num_layers,
                    hidden_sz, num_dir, emb_dim, embfix, pretr_emb, dropout, mf)))
        results += ('\nEPOCH{:2d} - loss: {:.4f}  acc: {:6.4f}%({:3d}/{}) t5_acc: {:6.4f}%({:3d}' \
                '/{}) MRR: {:.6f}'.format(epoch, avg_loss, accuracy,
                                        corrects, size, t5_acc, t5_corrects, size,
                                        mrr))

    print(highest_t1_acc_metrics + '\n')
    writeResults(args, results, highest_t1_acc, highest_t1_acc_metrics, highest_t1_acc_params)

def writeResults(args, results, highest_t1_acc, highest_t1_acc_metrics, highest_t1_acc_params):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    f = open(save_path + '/results.txt','w')
    f.write('PARAMETERS:\n' \
            'Net Type: %s\n' \
            #'Learning Rate: %f\n' \
            'Epochs: %i\n' \
            'Batch Size: %i\n' \
            'Optimizer: %s\n' \
            'Num Layers: %i\n' \
            'Hidden Size: %i\n' \
            'Num Directions: %i\n'
            'Embedding Dimension: %i\n' \
            'Fixed Embeddings: %s\n' \
            'Pretrained Embeddings: %s\n' \
            'Dropout: %.1f\n' \
            'Min Freq: %d'
            % (net_type, epochs, bs, opt, num_layers,
            hs, num_dir, emb_dim, embfix, pretr_emb, dropout, mf))
    f.write(results)
    f.close()
    if highest_t1_acc > acc_thresh:
        g = open(save_path + folder+ '/best_models.txt','a')
        g.write(highest_t1_acc_metrics)
        g.write(highest_t1_acc_params)
        g.close()

def parseParams():
    parser = argparse.ArgumentParser(description='LSTM text classifier')
    # data
    parser.add_argument('-data-path', type=str, default='../new_data/', help='data path [default: ../new_data/]') #
    parser.add_argument('-train-path', type=str, default='kdata_train.tsv', help='data path [default: kdata_train.tsv]') #
    parser.add_argument('-dev-path', type=str, default='kdata_dev.tsv', help='data path [default: kdata_dev.tsv]') #
    parser.add_argument('-test-path', type=str, default='../basic/test.tsv', help='data path [default: kdata_test.tsv]') #

    # learning
    parser.add_argument('-mf', type=int, default=1, help='min_freq for vocab [default: 1]') #
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]') #
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 100]') #
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]') #
    parser.add_argument('-opt', type=str, default='adamax', help='optimizer [default: adamax]') #

    # model
    parser.add_argument('-net-type', type=str, default='lstm', help='network type [default: lstm]')
    parser.add_argument('-num-layers', type=int, default=4, help='number of layers [default: 1]') #
    parser.add_argument('-hidden-sz', type=int, default=500, help='hidden size [default: 300]') #
    parser.add_argument('-num-dir', type=int, default=2, help='number of directions [default: 2]') #
    parser.add_argument('-emb-dim', type=int, default=300, help='number of embedding dimension [default: 300]') #
    parser.add_argument('-embfix', type=str, default=False, help='fix the embeddings [default: False]') #
    parser.add_argument('-pretr-emb', type=str, default=False, help='use pretrained embeddings') #
    parser.add_argument('-dropout', type=float, default=.5, help='dropout rate [default: .5]')
    parser.add_argument('-pred-filter', type=bool, default=True, help='Filter preds using SNI [default: True]')

    # options
    parser.add_argument('-save-path', type=str, default='./saved_models', help='path to save models [default: ./saved_models]')
    parser.add_argument('-save', type=bool, default=False, help='save model [default: False]')
    parser.add_argument('-folder', type=str, default='', help='folder to save models [default: '']')
    parser.add_argument('-acc-thresh', type=float, default=40, help='top1 accuracy threshold to save model')
    parser.add_argument('-device', type=int, default=0, help='GPU to use [default: 0]')
    args = parser.parse_args()

    args.embfix = (args.embfix == 'True')
    args.pretr_emb = (args.pretr_emb == 'True')

    args.save_path_full = args.save_path + \
                        args.folder + \
                        '/net-' + str(args.net_type) + \
                        '_e' + str(args.epochs) + \
                        '_bs' + str(args.batch_size) + \
                        '_opt-' + str(args.opt) + \
                        '_ly' + str(args.num_layers) + \
                        '_hs' + str(args.hidden_sz) + \
                        '_dr' + str(args.num_dir) + \
                        '_ed' + str(args.emb_dim) + \
                        '_femb' + str(args.embfix) + \
                        '_ptemb' + str(args.pretr_emb) + \
                        '_drp' + str(args.dropout)
    if args.mf > 1: args.save_path_full += '_mf' + str(args.mf)
    return args

if __name__ == '__main__':
    main()
