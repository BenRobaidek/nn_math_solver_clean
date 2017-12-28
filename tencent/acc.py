import numpy as np
with open('pred_folds_rm.txt') as f:
    preds = np.array(f.readlines())
with open('./data/tgt-test.txt') as f:
    tgts = np.array(f.readlines())
print(np.sum(tgts[:] == preds[:])/len(preds))
