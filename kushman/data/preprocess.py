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
import sympy

sys.path.append('../sni_saved_models/')
sys.path.append('../../sni/model/')
import model
import evalTest

def main():

    # LOAD JSON DATA
    jsondata = json.loads(open('./input/Kushman.json').read())
    jsondata_no_sni = copy.deepcopy(jsondata)

    # LOAD SNI MODEL
    sni_model = torch.load('../sni_saved_models/best_model.pt')
    if int(torch.cuda.is_available()) == 1:
        sni_model = sni_model.cuda()
    print(sni_model)

    sni_model.lstm.flatten_parameters()
    sni_model.eval()
    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABEL = data.Field(sequential=False)

    fields = [('text', TEXT), ('label', LABEL)]
    train = data.TabularDataset(path='./working/sni/train.tsv', format='tsv', fields=fields)
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    LABEL.build_vocab(train)

    # PREPROCESS DATA W/ SNI
    print('Preprocessing with sni...')
    for x in jsondata:
        lQueryVars = x.get('lQueryVars')
        x['sQuestion'], x['lEquations'], x['variables'] = preprocess(x['sQuestion'], x['lEquations'], lQueryVars, sni_model, fields, use_sni=True)
    print('Preprocessing with sni complete...')

    # PREPROCESS DATA WITHOUT SNI
    print('Preprocessing without sni...')
    for x in jsondata_no_sni:
        lQueryVars = x.get('lQueryVars')
        x['sQuestion'], x['lEquations'], x['variables'] = preprocess(x['sQuestion'], x['lEquations'], lQueryVars, sni_model, fields, use_sni=False)
    print('Preprocessing without sni complete...')

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

    # SAVE SRC/TGT FILES W/ SNI
    if not os.path.exists('./working/basic/'): os.makedirs('./working/basic/')

    json2tsv(train_indicesk123, jsondata,   './working/basic/traink1234.tsv')
    json2tsv(val_indicesk4,     jsondata,   './working/basic/valk1234.tsv')
    json2tsv(test_indicesk5,    jsondata,   './working/basic/testk5.tsv')

    json2tsv(train_indicesk234, jsondata,   './working/basic/traink2345.tsv')
    json2tsv(val_indicesk5,     jsondata,   './working/basic/valk2345.tsv')
    json2tsv(test_indicesk1,    jsondata,   './working/basic/testk1.tsv')

    json2tsv(train_indicesk345, jsondata,   './working/basic/traink3451.tsv')
    json2tsv(val_indicesk1,     jsondata,   './working/basic/valk3451.tsv')
    json2tsv(test_indicesk2,    jsondata,   './working/basic/testk2.tsv')

    json2tsv(train_indicesk451, jsondata,   './working/basic/traink4512.tsv')
    json2tsv(val_indicesk2,     jsondata,   './working/basic/valk4512.tsv')
    json2tsv(test_indicesk3,    jsondata,   './working/basic/testk3.tsv')

    json2tsv(train_indicesk512, jsondata,   './working/basic/traink5123.tsv')
    json2tsv(val_indicesk3,     jsondata,   './working/basic/valk5123.tsv')
    json2tsv(test_indicesk4,    jsondata,   './working/basic/testk4.tsv')

    # SAVE SRC/TGT FILES W/O SNI
    if not os.path.exists('./working/no_sni/'): os.makedirs('./working/no_sni/')

    json2tsv(train_indicesk123, jsondata_no_sni,   './working/no_sni/traink1234.tsv')
    json2tsv(val_indicesk4,     jsondata_no_sni,   './working/no_sni/valk1234.tsv')
    json2tsv(test_indicesk5,    jsondata_no_sni,   './working/no_sni/testk5.tsv')

    json2tsv(train_indicesk234, jsondata_no_sni,   './working/no_sni/traink2345.tsv')
    json2tsv(val_indicesk5,     jsondata_no_sni,   './working/no_sni/valk2345.tsv')
    json2tsv(test_indicesk1,    jsondata_no_sni,   './working/no_sni/testk1.tsv')

    json2tsv(train_indicesk345, jsondata_no_sni,   './working/no_sni/traink3451.tsv')
    json2tsv(val_indicesk1,     jsondata_no_sni,   './working/no_sni/valk3451.tsv')
    json2tsv(test_indicesk2,    jsondata_no_sni,   './working/no_sni/testk2.tsv')

    json2tsv(train_indicesk451, jsondata_no_sni,   './working/no_sni/traink4512.tsv')
    json2tsv(val_indicesk2,     jsondata_no_sni,   './working/no_sni/valk4512.tsv')
    json2tsv(test_indicesk3,    jsondata_no_sni,   './working/no_sni/testk3.tsv')

    json2tsv(train_indicesk512, jsondata_no_sni,   './working/no_sni/traink5123.tsv')
    json2tsv(val_indicesk3,     jsondata_no_sni,   './working/no_sni/valk5123.tsv')
    json2tsv(test_indicesk4,    jsondata_no_sni,   './working/no_sni/testk4.tsv')

def preprocess(question, equation, lQueryVars, sni_model, fields, use_sni):
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

    # prepend and postpend null tokens to question to allow for sni window size
    # of three
    question = ['null', 'null', 'null'] + question + ['null', 'null', 'null']

    # prevent inplace changes on question
    question_copy = [t for t in question]
    # find and replace constants in question and equation
    i = 0
    constants = dict()
    for j,token in enumerate(question):
        if isFloat(token):
            example = question_copy[j-3:j+4]
            ex = data.Example.fromlist([' '.join(example), ''], fields)
            dataset = data.Dataset([ex], fields)
            inp = None
            iterator = data.Iterator(dataset, batch_size=1)
            iterator.repeat=False
            for batch in iterator:
                inp = batch.text.t()#.cuda()
                #inp = inp.cuda(device=0)

            if (not use_sni) or (use_sni and isSignificant(inp, sni_model)):
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

    # simplify equation
    print('equation (before):', equation)
    equation = equation.split(',')
    for x in equation:
        x = x.split('=')
        x = '(' + x[0] + ')' + '-' + '(' + x[1] + ')'
        print('x:', x)
        print(sympy.simplify(x))

    print('equation (before):', equation)


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

def isSignificant(inp, sni_model):
    """
    Returns True iff inp is classified as significant by sni_model
    """
    return(evalTest.fast_test(inp, sni_model).data[0] == 1)

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
