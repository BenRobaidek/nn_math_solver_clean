import torch
from torch import autograd, nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../tencent/')
sys.path.append('../kushman/')
from tencent import solver as tencent_solver
from kushman import solver as kushman_solver

from py_expression_eval import Parser
import re


def evaluate(data_iter, model, TEXT, emb_dim, LABELS, VAR_VALUES, ANS, snis, pred_filter=True, solver=None):
    model.eval()
    corrects, true_corrects, answer_correspond_to_equation, avg_loss, t5_corrects, rr = 0, 0, 0, 0, 0, 0

    eval_preds = []
    for batch_count,batch in enumerate(data_iter):
        inp, target, var_values, ans = batch.text, batch.label, batch.var_values, batch.ans
        inp.data.t_()
        #print('batch.var_values', batch.var_values)

        logit = model(inp)

        #print('values:', values)

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

        # TRUE ACCURACY
        equations = np.array(LABELS.vocab.itos)[np.array(preds.data)]
        variables = np.array(VAR_VALUES.vocab.itos)[np.array(batch.var_values.data)]
        answers = np.array(ANS.vocab.itos)[np.array(batch.ans.data)]

        # solve predicted equations if possible, true iff solved correctly
        pred_corrects = solver.solve(equations, variables, answers)
        true_corrects += np.sum(pred_corrects)

        # solve tgt equations if possible, true iff solved correctly
        equations = np.array(LABELS.vocab.itos)[np.array(batch.label.data)]
        tgt_corrects = solver.solve(equations, variables, answers)
        answer_correspond_to_equation += np.sum(tgt_corrects)

        # get classifier probabilities
        probabilities,_ = torch.max(F.softmax(logit, dim=1), dim=1)

        # append correct and probability
        result = []
        for pred_correct, probability in zip(pred_corrects, probabilities):
            result = np.append(result, [str(pred_correct) + '\t' + str(probability.data[0])])
        eval_preds = np.append(eval_preds, [result])

        # RANK 5
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
    true_acc = 100.0 * true_corrects/size
    #print('acc:', accuracy)
    #print('true_acc:', true_acc)
    #print('answer_correspond_to_equation:', answer_correspond_to_equation/size)
    #print()
    t5_acc = 100.0 * t5_corrects/size
    mrr = rr/size
    model.train()
    return(avg_loss, accuracy, true_acc, corrects, size, t5_acc, t5_corrects, mrr, eval_preds)
