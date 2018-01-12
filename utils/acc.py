from py_expression_eval import Parser
import numpy as np
import copy

with open('../tests/train_best_models/saved_models/classifier_basic/answers.txt') as f:
    preds = np.array(f.readlines())
with open('../tencent/data/working/basic/answers.txt') as f:
    tgts = np.array(f.readlines())
parser = Parser()
for i,line in enumerate(preds):
    output = None
    try:
        output = float(line)
        output = parser.evaluate(output, variables=None)
    except (ValueError,TypeError):
        output = output
    preds[i] = output

for i,line in enumerate(tgts):
    output = None
    try:
        output = float(line)
        output = parser.evaluate(output, variables=None)
    except (ValueError,TypeError):
        output = output
    tgts[i] = output
print(preds)
print(tgts)

corrects = copy.deepcopy(preds)

for i,example in enumerate(corrects):
    try: 
        float(preds[i])
        float(tgts[i])
        if (float(preds[i]) - float(tgts[i]) / float(tgts[i]) <= .02):
            corrects[i] = 1
        else:
            corrects[i] = 0
    except (ValueError, ZeroDivisionError):
        corrects[i] = 0
corrects = np.array(corrects).astype(int)
print(np.sum(corrects) / len(corrects))
for i,c in enumerate(corrects):
    if c == 1:
        print('tgt:', tgts[i], 'pred:', preds[i])
#print(np.sum(tgts[:] == preds[:])/len(preds))
