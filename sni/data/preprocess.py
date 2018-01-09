import json
import numpy as np
import random
import math
import re
import sys
sys.path.append('../../tencent/data/')
from preprocess

def main():

    # LOAD DATA
    data = json.loads(open('../../tencent/data/Math23K.json').read())

    # PREPROCESS DATA
    for d in data:
        d['examples'] = preprocess(d['segmented_text'], d['equation'])

    # 5 FOLD CROSS VALIDATION
    print('Using existing cross validation splits')
    #print('Preforming cross validation splits...')
    #crossValidation(data, k = 5, k_test=5)

    # SAVE SPLIT INDICES
    split('./Math23K-train.txt', './Math23K-dev.txt', './Math23K-test.txt', k_test=5)

    # SAVE SRC/TGT files
    train_indices = np.genfromtxt('./Math23K-train.txt').astype(int)
    dev_indices = np.genfromtxt('./Math23K-dev.txt').astype(int)
    test_indices = np.genfromtxt('./Math23K-test.txt').astype(int)
    json2txt(train_indices, data,   './train.tsv')
    json2txt(dev_indices,   data,   './val.tsv')
    json2txt(test_indices,  data,   './test.tsv')


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
    equation = np.array([token.strip() for token in equation])

    examples = []

    for i,number in enumerate(numbers):
        if number.strip() in equation:
            index = np.where(question == number)[0][0]
            src = question[index-3:index+4]
            src = ' '.join(src)
            #print('np.shape(examples):', np.shape(examples))
            examples = np.append(examples, [src + '\t' + 'yes'])
        else:
            index = np.where(question == number)[0][0]
            src = question[index-3:index+4]
            src = ' '.join(src)
            examples = np.append(examples, [src + '\t' + 'no'])
    return examples

def json2txt(json_indices, data, output_path):
    output = open(output_path, 'w')
    for d in data:
        if int(d['id']) in json_indices:
            print(d['examples'])
            for example in d['examples']:
                print(example)
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
