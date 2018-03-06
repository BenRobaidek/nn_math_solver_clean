import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
import re
import sys
sys.path.append('../')
sys.path.append('../tencent/')
from tencent import solver as tencent_solver

def main():
    train(data_path='../tencent/data/working/basic/',
            train_path='traink1234.tsv',
            val_path='valk1234.tsv',
            test_path='testk5.tsv')

def train(data_path, train_path, val_path, test_path,):
    print('Training...')
    # LOAD DATA
    train = np.array([x for x in tencentDataset(data_path+train_path)])
    val = [x for x in tencentDataset(data_path+val_path)]
    test = [x for x in tencentDataset(data_path+test_path)]

    print('Getting tfidfs for train...')
    train_tfidfs = [tfidfs(x.split(),[x[0] for x in train]) for x in [x[0] for x in train]]
    print('Done...')

    corrects = 0
    pred_equations = np.array([])
    for x in val:
        problem_tfidfs = tfidfs(x[0].split(), [x[0] for x in train])
        if x[1] == train[getClosestIndex(problem_tfidfs, train_tfidfs)][1]:
            corrects += 1
        pred_equations = np.append(pred_equations,train[getClosestIndex(problem_tfidfs, train_tfidfs)][1])
        print('gold:', x[1], '\npred:', train[getClosestIndex(problem_tfidfs, train_tfidfs)][1])

    print(pred_equations)
    print('classification accuracy (VAL):', corrects/len([x[0] for x in val]))

    variables = [x[2] for x in val]
    answers = [x[3] for x in train]
    pred_corrects = tencent_solver.solve(pred_equations, variables, answers)
    print('true_acc (VAL):', sum(pred_corrects)/len([x[0] for x in val]))


def getClosestIndex(problem_tfidfs, train_tfidfs):
    return train_tfidfs.index(max(train_tfidfs, key=lambda x: jaccard(problem_tfidfs,x)))

def tfidfs(problem, document):
    return [problem.count(x)*idf(x,document) for x in problem]

def idf(word, document):
    #print('word:', word)
    D = len(document)
    occurences = 0
    for x in document:
        if word in [x.strip() for x in x.split()]:
            occurences += 1
    idf = None
    if occurences == 0:
        #print('idf:', 'inf')
        return float('inf')
    else:
        #print('idf:', D/occurences)
        return D/occurences

def jaccard(A,B):
    return sum([min(x_i,y_i) for x_i,y_i in zip(A,B)])/sum([max(x_i,y_i) for x_i,y_i in zip(A,B)])

def tencentDataset(path):
    data = np.array([x.split('\t') for x in open(path).readlines()])
    for i,x in enumerate(data):
        x[3] = x[3].strip().replace('%', '*.01')
        x[3] = re.sub(r'([\d])([(])', r'\1/\2', x[3])
        x[3] = eval(x[3])
    return data

if __name__ == '__main__':
    main()
