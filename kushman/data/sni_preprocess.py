import json
import numpy as np
import random
import math
import re
import sys
from py_expression_eval import Parser
import os

def main():

    # LOAD DATA
    data = json.loads(open('./input/Kushman.json').read())
    print(data)

    # PREPROCESS DATA
    for d in data:
        d['examples'] = preprocess(d['sQuestion'], ' '.join(d['lEquations']))

    # 5 FOLD CROSS VALIDATION
    print('Using existing cross validation splits')
    #print('Preforming cross validation splits...')
    #crossValidation(data, k = 5, k_test=5)

    # SAVE SPLIT INDICES
    train_indices, val_indices, test_indices = split_indices(k_test=5)

    # SAVE SRC/TGT files
    if not os.path.exists('./working/'): os.makedirs('./working/')
    if not os.path.exists('./working/sni/'): os.makedirs('./working/sni/')
    json2txt(train_indices, data,   './working/sni/train.tsv')
    json2txt(val_indices,   data,   './working/sni/val.tsv')
    json2txt(test_indices,  data,   './working/sni/test.tsv')


def crossValidation(data, k = 5, k_test=5):
    # Saves k folds
    # k: k fold cross validation
    # k_test: fold to use for test

    random.shuffle(data)
    fold_size = math.floor(np.shape(data)[0] / k)
    for i in range(1, k + 1):
        output = open('fold' + str(i) + '.txt', 'w')
        for d in data[(i-1) * fold_size: i * fold_size]:
            output.write(d['id'] + '\n')
        output.close()
        print('fold' + str(i) + '.txt' + ' saved')

def split_indices(k_test=5):
    """
    Returns train, validation, and test indices
    foldi.txt files must already exist in ./input/
    """
    train_val = []
    for i in range(1,6):
        if not i == k_test:
            train_val = np.append(train_val, open('./input/fold' + str(i) + '.txt').readlines())
    #random.shuffle(train_val)
    test = open('./input/fold' + str(k_test) + '.txt').readlines()
    train_indices = np.array(train_val[0:-102]).astype(int)
    val_indices = np.array(train_val[-102:]).astype(int)
    test_indices = np.array(test).astype(int)
    return train_indices, val_indices, test_indices

def preprocess(question, equation):
    """
    Returns preprocessed version of question and equation
    """

    question = question.replace('. ', ' . ')
    question = question.replace('?', ' ? ')
    question = question.split()
    question = np.append(['null', 'null', 'null'], question)
    question = np.append(question, ['null', 'null', 'null'])
    question = [float(token) if isFloat(token) else token for token in question]

    numbers = np.array([token for token in question if isFloat(token)])

    _, indices = np.unique(numbers, return_index=True)
    numbers = numbers[np.sort(indices)]

    equation = equation.replace(',', ' , ')
    equation = equation.replace('+', ' + ')
    equation = equation.replace('-', ' - ')
    equation = equation.replace('*', ' * ')
    equation = equation.replace('/', ' / ')
    equation = equation.replace('^', ' ^ ')
    equation = equation.replace('**', ' ** ')
    equation = equation.replace('(', ' ( ')
    equation = equation.replace(')', ' ) ')
    equation = equation.replace('=', ' = ')
    equation = equation.split()

    equation = [float(token) if isFloat(token) else token for token in equation]

    examples = []

    for i,token in enumerate(question):
        if isFloat(token) and token in numbers and token in equation:
            src = ' '.join([str(x) for x in question[i-3:i+4]])
            examples = np.append(examples, [src + '\t' + 'yes'])
            #print('example:', src + '\t' + 'yes')
        elif isFloat(token) and token in numbers and not token in equation:
            src = ' '.join([str(x) for x in question[i-3:i+4]])
            examples = np.append(examples, [src + '\t' + 'no'])
            #print('example:', src + '\t' + 'no')
    return examples

def json2txt(json_indices, data, output_path):
    output = open(output_path, 'w')
    for d in data:
        if int(d['iIndex']) in json_indices:
            #print(d['examples'])
            for example in d['examples']:
                #print(example)
                output.write(example + '\n')
    output.close()

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

if __name__ == '__main__':
    main()
