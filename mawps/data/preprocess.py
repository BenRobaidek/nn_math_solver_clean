import json
import re
from sklearn.metrics import jaccard_similarity_score
import numpy as np

def main():
    # LOAD JSON DATA
    jsondata = json.loads(open('./input/AllWithEquations.json').read())

    val_indices = getIndices(jsondata, [x.split('\t')[0] for x in open('./val.tsv').readlines()])
    assert len(open('./val.tsv').readlines()) == len(val_indices)
    output = open('./val_indices.txt', 'w')
    for line in val_indices:
        output.write(str(line))
    output.close()

    train_indices = getIndices(jsondata, [x.split('\t')[0] for x in open('./train.tsv').readlines()])
    assert len(open('./train.tsv').readlines()) == len(train_indices)
    output = open('./train_indices.txt', 'w')
    for line in train_indices:
        output.write(str(line))
    output.close()

    test_indices = getIndices(jsondata, [x.split('\t')[0] for x in open('./test.tsv').readlines()])
    assert len(open('./test.tsv').readlines()) == len(test_indices)
    output = open('./test_indices.txt', 'w')
    for line in test_indices:
        output.write(str(line))
    output.close()



def getIndices(jsondata, problems):
    results = []
    for i,x in enumerate(problems):
        print(x)
        x = [x.lower() for x in x.split()]
        y = [[x.lower() for x in d['sQuestion'].replace('?', ' ? ').split()] for d in jsondata]
        jaccards = [jaccard(x,z) for z in y]
        print(jsondata[np.argmax(jaccards)]['sQuestion'])
        results = np.append(results, [jsondata[np.argmax(jaccards)]['iIndex']])
        print()
    return results

def jaccard(a, b):
    c = set(a).intersection(set(b))
    return (float(len(c)) / (len(set(a)) + len(set(b)) - len(c))) * (1 - (abs(len(a) - len(b)) / (len(a) + len(b))))




def isFloat(value):
    """
    Returns True iff value can be represented as a float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    main()
