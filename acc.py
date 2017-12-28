import numpy as np
with open('pred.txt') as f:
    preds = np.array(f.readlines())
with open('./data/tgt-val.txt') as f:
    tgts = np.array(f.readlines())
print(np.sum(tgts[:] == preds[:])/len(preds))
