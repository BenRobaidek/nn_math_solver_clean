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
        lQueryVars = x.get('lQueryVars')
        x['sQuestion'], x['lEquations'], x['variables'] = preprocess(x['sQuestion'], x['lEquations'], lQueryVars)

    # 5 FOLD CROSS VALIDATION
    print('Using existing cross validation splits')
    # use the code below to generate new folds
    #print('Preforming cross validation splits...')
    #crossValidation(jsondata, k = 5, k_test=5)

    # GET TRAIN, VAL, TEST INDICES
    train_indicesk123, val_indicesk4, test_indicesk5 = split_indices_crossval(k_val=4, k_test=5)
    train_indicesk234, val_indicesk5, test_indicesk1 = split_indices_crossval(k_val=5, k_test=1)
    train_indicesk345, val_indicesk1, test_indicesk2 = split_indices_crossval(k_val=1, k_test=2)
    train_indicesk451, val_indicesk2, test_indicesk3 = split_indices_crossval(k_val=2, k_test=3)
    train_indicesk512, val_indicesk3, test_indicesk4 = split_indices_crossval(k_val=3, k_test=4)

    # SAVE SRC/TGT FILES
    if not os.path.exists('./working/basic/'): os.makedirs('./working/basic/')

    json2tsv(train_indicesk123, jsondata,   './working/basic/traink123.tsv')
    json2tsv(val_indicesk4,     jsondata,   './working/basic/valk4.tsv')
    json2tsv(test_indicesk5,    jsondata,   './working/basic/testk5.tsv')

    json2tsv(train_indicesk234, jsondata,   './working/basic/traink234.tsv')
    json2tsv(val_indicesk5,     jsondata,   './working/basic/valk5.tsv')
    json2tsv(test_indicesk1,    jsondata,   './working/basic/testk1.tsv')

    json2tsv(train_indicesk345, jsondata,   './working/basic/traink345.tsv')
    json2tsv(val_indicesk1,     jsondata,   './working/basic/valk1.tsv')
    json2tsv(test_indicesk2,    jsondata,   './working/basic/testk2.tsv')

    json2tsv(train_indicesk451, jsondata,   './working/basic/traink451.tsv')
    json2tsv(val_indicesk2,     jsondata,   './working/basic/valk2.tsv')
    json2tsv(test_indicesk3,    jsondata,   './working/basic/testk3.tsv')

    json2tsv(train_indicesk512, jsondata,   './working/basic/traink512.tsv')
    json2tsv(val_indicesk3,     jsondata,   './working/basic/valk3.tsv')
    json2tsv(test_indicesk4,    jsondata,   './working/basic/testk4.tsv')

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
    variables = [x for x in equation if x not in ['+', '-', '*', '/', ',',
            '**', '(', ')', '='] and not isFloat(x) and not re.match(r'\[[a-z]\]', x)]
    variables = np.unique(variables)
    i = 0
    for v in variables:
        equation = [x if x!=v else 'VAR_' + str(i) for x in equation]
        i += 1

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
            solutions =  ','.join([str(x) for x in d['lSolutions']])
            output.write(str(d['sQuestion']) + '\t' +
                    str(d['lEquations']) + '\t' + str(d['variables'])
                    + '\t' + '[' + solutions + ']' + '\n')
    output.close()

def split_indices_crossval(k_val=4, k_test=5):
    """
    Returns train, validation, and test indices
    foldi.txt files must already exist in ./input/
    """
    train = []
    val = []
    test = []
    for i in range(1,6):
        if i == k_test:
            test = np.append(test, open('./input/fold' + str(i) + '.txt').readlines())
        elif i == k_val:
            val = np.append(val, open('./input/fold' + str(i) + '.txt').readlines())
        else:
            train = np.append(train, open('./input/fold' + str(i) + '.txt').readlines())
    train_indices = np.array(train).astype(int)
    val_indices = np.array(val).astype(int)
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