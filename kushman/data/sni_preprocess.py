import json
import numpy as np
import random
import math
import re
import sys
from py_expression_eval import Parser

def main():

    # LOAD DATA
    data = json.loads(open('./input/Math23K.json').read())

    # PREPROCESS DATA
    for d in data:
        d['examples'] = preprocess(d['segmented_text'], d['equation'])

    # 5 FOLD CROSS VALIDATION
    print('Using existing cross validation splits')
    #print('Preforming cross validation splits...')
    #crossValidation(data, k = 5, k_test=5)

    # SAVE SPLIT INDICES
    train_indices, val_indices, test_indices = split_indices(k_test=5)

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
    train_indices = np.array(train_val[0:-1000]).astype(int)
    val_indices = np.array(train_val[-1000:]).astype(int)
    test_indices = np.array(test).astype(int)
    return train_indices, val_indices, test_indices

def split(train_path, dev_path, test_path, k_test=5):
    train_dev = []
    for i in range(1,6):
        if not i == k_test:
            train_dev = np.append(train_dev, open('fold' + str(i) + '.txt').readlines())
    #random.shuffle(train_dev)
    test = open('fold' + str(k_test) + '.txt').readlines()

    # Train
    output = open(train_path, 'w')
    for d in train_dev[0:-1000]:
        output.write(d)
    output.close()
    print(train_path + ' saved')

    # Dev
    output = open(dev_path, 'w')
    for d in train_dev[-1000:]:
        output.write(d)
    output.close()
    print(dev_path + ' saved')

    # Test
    output = open(test_path, 'w')
    for d in test:
        output.write(d)
    output.close()
    print(test_path + ' saved')

def preprocess(question, equation):
    """
    Returns preprocessed version of question and equation
    """
    # handle %'s
    question = question.replace('%', ' % ')

    # handle fractions
    parser = Parser()
    fractions = re.findall('\(\d+\)/\(\d+\)', question)
    fractions = np.append(fractions, re.findall('\(\d+/\d+\)', question))
    for i,fraction in enumerate(fractions):
        #question = question.replace(fraction, str(sys.maxsize - i))
        #equation = equation.replace(fraction, str(sys.maxsize - i))
        question = question.replace(fraction, str(parser.evaluate(fraction, variables=None)))
        equation = equation.replace(fraction, str(parser.evaluate(fraction, variables=None)))

    # handle numbers with units
    question = re.sub(r'(\d+)([A-z]{1,2})', r'\1 \2', question)

    # seperate equation at operators
    print('equation (before):', equation)
    equation = equation.replace('[', ' ( ')
    equation = equation.replace(']', ' ) ')
    equation = equation.replace('+', ' + ')
    equation = equation.replace('+', ' + ')
    equation = equation.replace('-', ' - ')
    equation = equation.replace('*', ' * ')
    equation = equation.replace('/', ' / ')
    equation = equation.replace('(', ' ( ')
    equation = equation.replace(')', ' ) ')
    equation = equation.replace('=', ' = ')
    equation = equation.replace('^', ' ^ ')


    # reduce %'s
    equation = equation.replace('%', ' / 100 ')

    # Preprocess Question

    question = question.split()
    question = np.append(['null', 'null', 'null'], question)
    question = np.append(question, ['null', 'null', 'null'])

    numbers = np.array([token for token in question if isFloat(token)])# or float(token) == 2)])
    _, indices = np.unique(numbers, return_index=True)
    numbers = numbers[np.sort(indices)]

    equation = np.array([token.strip() for token in equation.split(' ')])

    examples = []
    print('equation:', equation)

    for i,number in enumerate(numbers):
        index = np.where(question == number)[0][0]
        src = question[index-3:index+4]
        src = ' '.join(src)
        if number.strip() in equation:
            examples = np.append(examples, [src + '\t' + 'yes'])
            print('example:', src + '\t' + 'yes')
        else:
            examples = np.append(examples, [src + '\t' + 'no'])
            print('example:', src + '\t' + 'no')
    return examples

def json2txt(json_indices, data, output_path):
    output = open(output_path, 'w')
    for d in data:
        if int(d['id']) in json_indices:
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
