import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
import re

def main():
    train(data_path='../tencent/data/working/basic/',
            train_path='valk1234.tsv',
            val_path='valk1234.tsv',
            test_path='valk1234.tsv')

def train(data_path, train_path, val_path, test_path,):
    print('Training...')
    # LOAD DATA
    train = tencentDataset(data_path+train_path)
    val = tencentDataset(data_path+val_path)
    test = tencentDataset(data_path+test_path)

    print(tfidf('大米', train))

def tfidf(word, document):
    D = len(document)
    occurences = 0
    for x in document:
        if word in x[0].split(' '):
            occurences += 1

    print('D:', D)
    print('occurences:', occurences)
    return D/occurences

def jaccard(A,B):
    return len(A.intersection(B))/len(A.union(B))

def tencentDataset(path):
    data = np.array([x.split('\t') for x in open(path).readlines()])
    for i,x in enumerate(data):
        x[3] = x[3].strip().replace('%', '*.01')
        x[3] = re.sub(r'([\d])([(])', r'\1/\2', x[3])
        x[3] = eval(x[3])
    return data

if __name__ == '__main__':
    main()
