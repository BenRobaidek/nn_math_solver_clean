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
        output.write(str(line) + '\n')
    output.close()

    """
    train_indices = getIndices(jsondata, [x.split('\t')[0] for x in open('./train.tsv').readlines()])
    assert len(open('./train.tsv').readlines()) == len(train_indices)
    output = open('./train_indices.txt', 'w')
    for line in train_indices:
        output.write(str(line) + '\n')
    output.close()

    test_indices = getIndices(jsondata, [x.split('\t')[0] for x in open('./test.tsv').readlines()])
    assert len(open('./test.tsv').readlines()) == len(test_indices)
    output = open('./test_indices.txt', 'w')
    for line in test_indices:
        output.write(str(line) + '\n')
    output.close()
    """

    train = open('./train.tsv').readlines()
    val = open('./val.tsv').readlines()
    test = open('./test.tsv').readlines()

    #train_indices = open('./train_indices.txt').readlines()
    val_indices = open('./val_indices.txt').readlines()
    #test_indices = open('./test_indices.txt').readlines()

    for i,x in zip(val_indices,val):
        question = x.split('\t')[0].split()
        equation = x.split('\t')[1]
        var = None
        ans = None
        json_question = None
        for d in jsondata:
            if float(d['iIndex']) == float(i.strip()):
                ans = float(d['lSolutions'][0])
                json_question = d['sQuestion'].replace('?', ' ? ').replace('. ', ' . ').replace(',', ' , ').split()
                json_question = [x.lower() for x in json_question]


        print(question)
        print(json_question)
        assert(len(question) == len(json_question))
        #print(question, equation, var, ans)
        print()

    """
    for d in jsondata:
        d = preprocess(d)

    json2tsv(train_indices, jsondata)
    json2tsv(val_indices, jsondata)
    json2tsv(val_indices, jsondata)
    """




def preprocess(d):
    print(d)

def getIndices(jsondata, problems):
    results = []
    for i,question in enumerate(problems):
        question = [x.lower() for x in question.split()]
        json_questions = [[x.lower() for x in d['sQuestion'].replace('?', ' ? ').replace('. ', ' . ').replace(',', ' , ').split()] for d in jsondata]
        print('question:', question)
        best_json_question = [x for x in json_questions if looseEquals(x, question)]
        print('best_json_question:', best_json_question)
        assert len(best_json_question) == 1

        results = np.append(results, [])
        print()
    return results

def jaccard(a, b):
    c = set(a).intersection(set(b))
    return (float(len(c)) / (len(set(a)) + len(set(b)) - len(c))) * (1 - (abs(len(a) - len(b)) / (len(a) + len(b))))

def looseEquals(a,b):
    if not len(a) == len(b):
        return False
    else:
        allequal = True
        for A,B in zip(a,b):
            if not(A==B or '[' in A or '[' in B or isFloat(A) or isFloat(B)):
                allequal = False
        return allequal

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
