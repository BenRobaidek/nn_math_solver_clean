import torch
from torch import autograd, nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys

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

        # True Acc
        print('preds.data', preds.data)
        print('target.data', target.data)

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
