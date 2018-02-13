import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append('../../../../tencent/')
from tencent import solver

def main():
    printCorrects('./predk1.txt', '../basic/valk1.tsv', './correctsk1.tsv')
    printCorrects('./predk2.txt', '../basic/valk2.tsv', './correctsk2.tsv')
    printCorrects('./predk3.txt', '../basic/valk3.tsv', './correctsk3.tsv')
    printCorrects('./predk4.txt', '../basic/valk4.tsv', './correctsk4.tsv')
    printCorrects('./predk5.txt', '../basic/valk5.tsv', './correctsk5.tsv')

def printClassAcc(preds_path, tgts_path):
    preds = open(preds_path).readlines()
    tgts = open(tgts_path).readlines()
    corrects = 0
    for pred, tgt in zip(preds, tgts):

        if pred.strip() == tgt.strip():
            corrects += 1
        #else:
        #    print('pred:', pred.strip(), 'tgt:', tgt.strip())

    print('class accuracy:', corrects/len(tgts))

def printCorrects(equations_path, other_path, save_path):
    equations = open(equations_path).readlines()
    variables = [x.split('\t')[2] for x in open(other_path).readlines()]
    answers = [x.split('\t')[3].strip() for x in open(other_path).readlines()]

    #print('len(equations)', len(equations))
    #print('len(variables)', len(variables))
    #print('len(answers)', len(answers))

    with open(save_path, 'w') as f:
        for x in solver.solve(equations, variables, answers):
            f.write(str(x) + '\n')

if __name__ == '__main__':
    main()
