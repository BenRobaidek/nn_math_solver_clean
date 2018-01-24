import torch
from torch import autograd, nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
from py_expression_eval import Parser
import re

def evaluate(data_iter, model, TEXT, emb_dim, LABELS, VAR_VALUES, ANS, snis, pred_filter=True):
    model.eval()
    corrects, true_corrects, answer_correspond_to_equation, avg_loss, t5_corrects, rr = 0, 0, 0, 0, 0, 0
    eval_preds = ['equation', 'prediction', 'target']

    for batch_count,batch in enumerate(data_iter):
        inp, target, var_values, ans = batch.text, batch.label, batch.var_values, batch.ans
        inp.data.t_()
        #print('batch.var_values', batch.var_values)

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
        predictions = np.array(LABELS.vocab.itos)[np.array(preds.data)]
        targets = np.array(LABELS.vocab.itos)[np.array(batch.label.data)]
        var_values = np.array(VAR_VALUES.vocab.itos)[np.array(batch.var_values.data)]
        answers = np.array(ANS.vocab.itos)[np.array(batch.ans.data)]

        for prediction, tgt, var_value, answer in zip(predictions, targets, var_values, answers):
            result = str(prediction)
            var_value = eval(var_value)
            # sub variables into predicted and target equations
            for k in var_value:
                prediction = prediction.replace(k, var_value[k])
                tgt = tgt.replace(k, var_value[k])

            # Add multiplication symbols to answer where needed
            answer = re.sub(r'\(\((\d+)\)/\((\d+)\)\)',r'(\1/\2)',answer)
            answer = re.sub(r'(\d)\(',r'\1+(', answer, 1)
            # replace % in answer
            answer = answer.replace('%', ' / 100')
            answer = eval(answer)

            # replace ^ with ** in predicted equation
            prediction = prediction.replace('^', '**')
            # replace ^ with ** in tgt equation
            tgt = tgt.replace('^', '**')
            # remove = from equations
            prediction = prediction.strip('x =')
            tgt = tgt.strip('x =')
            # evaluate
            prediction = prediction.strip()

            if (not prediction == '80千米 / 小时') and (not re.search(r'\[\S\]', prediction)) and (not prediction == '<unk>'):
                try:
                    #print('prediction:', prediction)
                    prediction = eval(prediction)
                except ZeroDivisionError:
                    pass

            if (tgt == '<unk>'):
                pass
            elif (not tgt == 'x = 80千米 / 小时'):
                tgt = eval(tgt)

            if (not prediction == '<unk>') and (not tgt == '<unk>'):
                try:
                    prediction = float(prediction)
                    tgt = float(tgt)
                    error = abs(prediction - tgt)
                    if error <= .002:
                        true_corrects += 1
                except Exception as e:
                    #print(e)
                    pass

            if tgt == '<unk>':
                answer_correspond_to_equation += 1
            elif (not tgt == '<unk>'):
                try:
                    error = abs(answer - tgt)
                    if error <= .002:
                        answer_correspond_to_equation += 1
                except Exception as e:
                    #print(e)
                    pass
            result = result + '\t' + str(prediction) + '\t' + str(tgt) + '\n'
            eval_preds = np.append(eval_preds, [result])

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
    true_acc = 100.0 * true_corrects/size
    print('acc:', accuracy)
    print('true_acc:', true_acc)
    print('answer_correspond_to_equation:', answer_correspond_to_equation/size)
    print()
    t5_acc = 100.0 * t5_corrects/size
    mrr = rr/size
    model.train()
    return(avg_loss, accuracy, true_acc, corrects, size, t5_acc, t5_corrects, mrr, eval_preds)
