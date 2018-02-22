import sys
import numpy as np
sys.path.append('../')
sys.path.append('../tencent')
from tencent import solver

def main():
    pred_equations = open('../tencent/data/output/cnn_basic/pred-testk5.txt').readlines()
    tgt_equations = [x.split('\t')[1] for x in open('../tencent/data/working/basic/testk5.tsv').readlines()]

    variables = [x.split('\t')[2] for x in open('../tencent/data/working/basic/testk5.tsv').readlines()]
    answers = [x.split('\t')[3] for x in open('../tencent/data/working/basic/testk5.tsv').readlines()]

    #print('pred_equations:', pred_equations)
    #print('tgt_equations:', tgt_equations)
    #print('variables:', variables)
    #print('answers:', answers)

    corrects = solver.solve(pred_equations, variables, answers)

    # Calculate per class accuracy
    class_corrects = 0
    assert len(pred_equations) == len(tgt_equations)
    for p,t in zip(pred_equations, tgt_equations):
        p = p.strip().split(';')
        t = t.strip().split(',')
        p = ','.join([x.strip() for x in p])
        t = ','.join([x.strip() for x in t])
        #print('p:', p)
        #print('t:', t)
        if not p.strip() == t.strip(): print(p.strip(),'_________',t.strip())
        if p.strip() == t.strip(): class_corrects += 1
    print('class accuracy:', 100*class_corrects/len(pred_equations))

    output = open('../tencent/data/output/cnn_basic/corrects_testk5.txt', 'w')
    print(corrects)
    for x in corrects:
        output.write(str(x) + '\n')
    output.close()

    print('True acc:', 100*np.sum(corrects)/len(corrects))

if __name__ == '__main__':
    main()
