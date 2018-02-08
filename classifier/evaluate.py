import torch
from torch import autograd, nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
from py_expression_eval import Parser
import re
import sympy
from sympy.parsing.sympy_parser import parse_expr

def evaluate(data_iter, model, TEXT, emb_dim, LABELS, VAR_VALUES, ANS, snis, pred_filter=True):
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


        # True Acc
        predictions = np.array(LABELS.vocab.itos)[np.array(preds.data)]
        targets = np.array(LABELS.vocab.itos)[np.array(batch.label.data)]
        var_values = np.array(VAR_VALUES.vocab.itos)[np.array(batch.var_values.data)]
        answers = np.array(ANS.vocab.itos)[np.array(batch.ans.data)]
        probabilities,_ = torch.max(F.softmax(logit), dim=1)

        for prediction, tgt, var_value, answer, probability in zip(predictions, targets, var_values, answers, probabilities):

            #print('prediction:', prediction)
            #print('tgt:', tgt)
            #print('var_value:', var_value)
            #print('answer:', answer)
            #print('probability:', probability)
            #print('answer:', answer)
            answer = eval(answer)
            for i,a in enumerate(answer):
                #print('type(a):', type(a))
                answer[i] = float(a)

            # sub variables into predicted and target equations
            var_value = eval(var_value)
            for k in var_value:
                prediction = prediction.replace(k, str(var_value[k]))
                tgt = tgt.replace(k, str(var_value[k]))

            # replace ^ with ** in predicted equation
            prediction = prediction.replace('^', '**')
            # replace ^ with ** in tgt equation
            tgt = tgt.replace('^', '**')
            # remove = from equations

            print('type(prediction):', type(prediction))
            #print('tgt:', tgt)

            if (prediction is not '<unk>') and '=' in prediction:

                # get variables out of predicted equation
                answer_variables = np.unique(re.findall(r'VAR_[\d]', prediction, flags=0))

                prediction = prediction.split(',')
                for k,p in enumerate(prediction):
                    prediction[k] = '(' + p.split('=')[1] + ') - (' + p.split('=')[0] + ')'

                #if len(prediction) == 1: prediction = prediction[0]
                answers = dict()
                print(answer_variables)

                if not len(np.unique(re.findall(r'\[[a-z]\]', ','.join(prediction)))) >= 1:
                    print('prediction:', prediction)
                    expr = sympy.Eq([parse_expr(x) for x in prediction])
                    symbols = sympy.symbols(' '.join(answer_variables))
                    print('expr:', expr)
                    print('symbols:', symbols)
                    answers = sympy.solveset(expr, symbols)
                    print('answers:', answers)

            """
            print('prediction:', prediction)
            print('tgt:', tgt)
            print('var_value:', var_value)
            print('answer:', answer)
            print('probability:', probability)
            """



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
    #print('answer_correspond_to_equation:', answer_correspond_to_equation/size)
    print()
    t5_acc = 100.0 * t5_corrects/size
    mrr = rr/size
    model.train()
    return(avg_loss, accuracy, true_acc, corrects, size, t5_acc, t5_corrects, mrr, eval_preds)
