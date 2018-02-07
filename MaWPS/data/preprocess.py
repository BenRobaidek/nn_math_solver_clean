import json
import copy
import numpy as np
import random
import math
import re
import sys
import os
import torch
from torchtext import data, datasets
from py_expression_eval import Parser

sys.path.append('../../sni/model')
import model
import evalTest

def main():


    # LOAD JSON DATA
    jsondata = json.loads(open('./input/Kushman.json').read())
    for x in jsondata:
        #print('BEFORE:', x['sQuestion'], x['lEquations'], x['lSolutions'])
        lQueryVars = x.get('lQueryVars')
        x['sQuestion'], x['lEquations'], x['variables'] = preprocess(x['sQuestion'], x['lEquations'], lQueryVars)
        if len(x['lEquations'].split(',')) >= 2:
            #print(x['lQueryVars'])
            print('AFTER: ', x['sQuestion'], x['lEquations'], x['lSolutions'], x['variables'])
            print()

    # 5 FOLD CROSS VALIDATION
    print('Using existing cross validation splits')
    # use the code below to generate new folds
    #print('Preforming cross validation splits...')
    #crossValidation(jsondata, k = 5, k_test=5)

    # GET TRAIN, VAL, TEST indices
    train_indices, val_indices, test_indices = split_indices(k_test=5)

    json2tsv(train_indices, jsondata, './working/train.tsv')
    json2tsv(val_indices, jsondata, './working/val.tsv')
    json2tsv(test_indices, jsondata, './working/test.tsv')

def preprocess(question, equation, lQueryVars):
    # handle $'s
    question = question.replace('$', ' $ ')
    question = question.replace('. ', ' . ')
    question = question.replace('?', ' ? ')
    question = re.sub(r',([\d\d\d])', r'\1', question)

    # join equations if needed
    equation = ' , '.join(equation)

    # seperate equation at operators
    equation = equation.replace('[', ' ( ')
    equation = equation.replace(']', ' ) ')
    equation = equation.replace('+', ' + ')
    equation = equation.replace('-', ' - ')
    equation = equation.replace('*', ' * ')
    equation = equation.replace('/', ' / ')
    equation = equation.replace('(', ' ( ')
    equation = equation.replace(')', ' ) ')
    equation = equation.replace('=', ' = ')
    equation = equation.replace('^', ' ^ ')

    equation = equation.split()
    question = question.split()

    # find and replace constants in question and equation
    i = 0
    constants = dict()
    for j,token in enumerate(question):
        if isFloat(token):
            token = float(token)
            character = '[' +chr(97 + i) + ']'
            for symbol in equation:
                if isFloat(symbol) and float(symbol) == float(token):
                    equation[equation.index(symbol)] = character
            constants[character] = str(token)
            for q in question:
                if isFloat(q) and float(q) == token:
                    question[question.index(q)] = character
            i += 1

    # find and replace variables in equation
    print('equation:', equation)
    variables = [x for x in equation if x not in ['+', '-', '*', '/', ',',
            '**', '(', ')', '='] and not isFloat(x) and not re.match(r'\[[a-z]\]', x)]
    variables = np.unique(variables)
    i = 0
    for v in variables:
        print(equation[str(equation) == v])
        i += 1
    print('equation:', equation)

    question = ' '.join(question)
    equation = ''.join(equation)
    #equation = equation.split(',')
    return question, equation, constants

def json2tsv(json_indices, json_data, output_path):
    """
    For each example in json_data indexed by json_indices,
    writes the associated question and equation to output_path
    """
    output = open(output_path, 'w')
    for d in json_data:
        if int(d['iIndex']) in json_indices:
            solutions =  ','.join(str(d['lSolutions']))
            output.write(str(d['sQuestion']) + '\t' +
                    str(d['lEquations']) + '\t' + str(d['variables'])
                    + '\t' + solutions + '\n')
    output.close()

def split_indices(k_test=5):
    """
    Returns train, validation, and test indices
    foldi.txt files must already exist in ./input/
    """
    train_val = []
    for i in range(1,6):
        if not i == k_test:
            train_val = np.append(train_val, open('./working/fold' + str(i) + '.txt').readlines())
    #random.shuffle(train_val)
    test = open('./working/fold' + str(k_test) + '.txt').readlines()
    train_indices = np.array(train_val[0:-1000]).astype(int)
    val_indices = np.array(train_val[-1000:]).astype(int)
    test_indices = np.array(test).astype(int)
    return train_indices, val_indices, test_indices

def crossValidation(data, k = 5, k_test=5):
    """
    Saves k folds from data
    k: k fold cross validation
    k_test: fold to use for test
    """
    random.shuffle(data)
    fold_size = math.floor(np.shape(data)[0] / k)
    for i in range(1, k + 1):
        output = open('./working/fold' + str(i) + '.txt', 'w')
        for d in data[(i-1) * fold_size: i * fold_size]:
            output.write(str(d['iIndex']) + '\n')
        output.close()
        print('fold' + str(i) + '.txt' + ' saved')

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
