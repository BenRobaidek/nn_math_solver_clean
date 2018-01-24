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
    jsondata = json.loads(open('./input/Math23K.json').read())
    jsondata_no_sni = copy.deepcopy(jsondata)

    # LOAD SNI MODEL
    model = torch.load('../../sni/saved_models/sni_best_model.pt')
    if int(torch.cuda.is_available()) == 1:
        model = model.cuda()
    print(model)

    model.lstm.flatten_parameters()
    model.eval()
    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABEL = data.Field(sequential=False)

    fields = [('text', TEXT), ('label', LABEL)]
    train = data.TabularDataset(path='../../sni/data/train.tsv', format='tsv', fields=fields)
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    LABEL.build_vocab(train)

    # LOAD EXISTING PREPROCESSED JSON DATA IF AVAILABLE
    #try:
    #    jsondata = json.loads(open('./working/Math23K-preprocessed.json').read())

    # PREPROCESS DATA
    print('Preprocessing...')
    for d in jsondata:
        #print('d[\'segmented_text\']:', d['segmented_text'])
        #print('d[\'equation\']', d['equation'])
        d['segmented_text'], d['equation'], d['variable_values'] = preprocess(d['segmented_text'], d['equation'], model, fields, use_sni=True)
        #print('d[\'segmented_text\']:', d['segmented_text'])
        #print('d[\'equation\']', d['equation'])
        #print()
    print('Preprocessing complete...')
    #print(jsondata)

    # PREPROCESS DATA WITHOUT SNI
    print('Preprocessing without sni...')
    for d in jsondata_no_sni:
        d['segmented_text'], d['equation'], d['variable_values'] = preprocess(d['segmented_text'], d['equation'], model, fields, use_sni=False)
        #print(d['segmented_text'])
    print('Preprocessing without sni complete...')

    # CREATE WORKING AND OUTPUT FOLDERS IF NEEDED
    if not os.path.exists('./working/'): os.makedirs('./working/')
    if not os.path.exists('./output/'): os.makedirs('./output/')

    # SAVE PREPROCESSED JSON
    #with open('./working/Math23K-preprocessed.json', 'w') as outfile:
    #    json.dump(jsondata, outfile)

    # 5 FOLD CROSS VALIDATION
    print('Using existing cross validation splits')
    # use the code below to generate new folds
    #print('Preforming cross validation splits...')
    #crossValidation(jsondata, k = 5, k_test=5)

    # GET TRAIN, VAL, TEST INDICES
    train_indices, val_indices, test_indices = split_indices(k_test=5)

    # SAVE SRC/TGT FILES
    if not os.path.exists('./working/basic/'): os.makedirs('./working/basic/')
    json2tsv(train_indices, jsondata,   './working/basic/train.tsv')
    json2tsv(val_indices,   jsondata,   './working/basic/val.tsv')
    json2tsv(test_indices,  jsondata,   './working/basic/test.tsv')

    # SAVE VARIABLE VALUES TO FILE
    saveValues(val_indices, jsondata,    './working/basic/val_values.txt')

    # SAVE SRC/TGT FILES NO SNI
    if not os.path.exists('./working/no_sni/'): os.makedirs('./working/no_sni/')
    json2tsv(train_indices, jsondata_no_sni,   './working/no_sni/train.tsv')
    json2tsv(val_indices,   jsondata_no_sni,   './working/no_sni/val.tsv')
    json2tsv(test_indices,  jsondata_no_sni,   './working/no_sni/test.tsv')

    # SAVE VARIABLE VALUES TO FILE NO SNI
    saveValues(val_indices, jsondata_no_sni,    './working/no_sni/val_values.txt')

    # REMOVE TEST FOLD BEFORE COUNTING UNCOMMON EQUATIONS
    jsondata = [d for d in jsondata if int(d['id']) not in test_indices]

    # REMOVE UNCOMMON EQUATIONS
    print('Removing uncommon equations...')
    print('Started with', len(jsondata), 'examples')
    common_data2, uncommon_data2 = mostCommon(jsondata, .2)
    common_data4, uncommon_data4 = mostCommon(jsondata, .4)
    common_data6, uncommon_data6 = mostCommon(jsondata, .6)
    common_data8, uncommon_data8 = mostCommon(jsondata, .8)
    #print('Filtered down to', len(common_data), 'examples')

    # SAVE TSV FILES (FILTERED DATA)
    if not os.path.exists('./working/common0.2/'): os.makedirs('./working/common0.2/')
    if not os.path.exists('./working/common0.4/'): os.makedirs('./working/common0.4/')
    if not os.path.exists('./working/common0.6/'): os.makedirs('./working/common0.6/')
    if not os.path.exists('./working/common0.8/'): os.makedirs('./working/common0.8/')

    # SAVE VARIABLE VALUES TO FILE FOR COMMON/UNCOMMON
    saveValues(val_indices, jsondata, './working/common0.2/val_values.txt')
    saveValues(val_indices, jsondata, './working/common0.4/val_values.txt')
    saveValues(val_indices, jsondata, './working/common0.6/val_values.txt')
    saveValues(val_indices, jsondata, './working/common0.8/val_values.txt')

    # SAVE ANSWER FILES
    """
    saveAnswers(val_indices, jsondata, './working/basic/answers.txt')
    saveAnswers(val_indices, jsondata, './working/common0.2/answers.txt')
    saveAnswers(val_indices, jsondata, './working/common0.4/answers.txt')
    saveAnswers(val_indices, jsondata, './working/common0.6/answers.txt')
    saveAnswers(val_indices, jsondata, './working/common0.8/answers.txt')
    saveAnswers(val_indices, jsondata, './working/no_sni/answers.txt')
    """

    train_val_indices = np.append(train_indices, val_indices)
    json2tsv(train_val_indices, common_data2,    './working/common0.2/train_val_common.tsv')
    json2tsv(train_val_indices, uncommon_data2,  './working/common0.2/train_val_uncommon.tsv')

    json2tsv(train_val_indices, common_data4,    './working/common0.4/train_val_common.tsv')
    json2tsv(train_val_indices, uncommon_data4,  './working/common0.4/train_val_uncommon.tsv')

    json2tsv(train_val_indices, common_data6,    './working/common0.6/train_val_common.tsv')
    json2tsv(train_val_indices, uncommon_data6,  './working/common0.6/train_val_uncommon.tsv')

    json2tsv(train_val_indices, common_data8,    './working/common0.8/train_val_common.tsv')
    json2tsv(train_val_indices, uncommon_data8,  './working/common0.8/train_val_uncommon.tsv')

    # SAVE FULL TSV FILES
    tsvs2tsv('./working/common0.2/train_val_common.tsv', './working/common0.2/train_val_uncommon.tsv', './working/common0.2/train_val.tsv')
    tsvs2tsv('./working/common0.4/train_val_common.tsv', './working/common0.4/train_val_uncommon.tsv', './working/common0.4/train_val.tsv')
    tsvs2tsv('./working/common0.6/train_val_common.tsv', './working/common0.6/train_val_uncommon.tsv', './working/common0.6/train_val.tsv')
    tsvs2tsv('./working/common0.8/train_val_common.tsv', './working/common0.8/train_val_uncommon.tsv', './working/common0.8/train_val.tsv')

    # SPLIT TRAIN VAL
    splitTrainVal('./working/common0.2/train_val.tsv', './working/common0.2/train.tsv', './working/common0.2/val.tsv')
    splitTrainVal('./working/common0.4/train_val.tsv', './working/common0.4/train.tsv', './working/common0.4/val.tsv')
    splitTrainVal('./working/common0.6/train_val.tsv', './working/common0.6/train.tsv', './working/common0.6/val.tsv')
    splitTrainVal('./working/common0.8/train_val.tsv', './working/common0.8/train.tsv', './working/common0.8/val.tsv')

    # REMOVE TEMPORARY FILES
    os.remove('./working/common0.2/train_val_common.tsv')
    os.remove('./working/common0.2/train_val_uncommon.tsv')
    os.remove('./working/common0.2/train_val.tsv')

    os.remove('./working/common0.4/train_val_common.tsv')
    os.remove('./working/common0.4/train_val_uncommon.tsv')
    os.remove('./working/common0.4/train_val.tsv')

    os.remove('./working/common0.6/train_val_common.tsv')
    os.remove('./working/common0.6/train_val_uncommon.tsv')
    os.remove('./working/common0.6/train_val.tsv')

    os.remove('./working/common0.8/train_val_common.tsv')
    os.remove('./working/common0.8/train_val_uncommon.tsv')
    os.remove('./working/common0.8/train_val.tsv')

    # SAVE VOCAB
    TEXT_class = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABEL_class = data.Field(sequential=False)
    fields_class = [('text', TEXT_class), ('label', LABEL_class)]
    train_class = data.TabularDataset(path='./working/basic/train.tsv', format='tsv', fields=fields_class)
    TEXT_class.build_vocab(train_class)
    LABEL_class.build_vocab(train_class)
    torch.save((list(TEXT_class.vocab.stoi), list(LABEL_class.vocab.stoi)), './working/basic/vocab.pt')

def crossValidation(data, k = 5, k_test=5):
    """
    Saves k folds from data
    k: k fold cross validation
    k_test: fold to use for test
    """
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

def mostCommon(data, percent):
    """
    Returns most common PERCENT of data by # of equation occurences
    uncommon 1-percent of data are returned as removed
    """
    equation, count= np.unique([d['equation'] for d in data], return_counts=True)
    indices = np.asarray((equation, count)).T[:,1].astype(int).argsort()
    result = np.asarray([[equation[i], count[i]] for i in indices])
    removed = np.array([])

    total_eqs = np.sum(np.asarray(result[:,1]).astype(int))
    occurences = 1
    while len(removed) < total_eqs * (1 - percent):
        print('Removing equations with', occurences, 'occurences...')
        equations_to_remove = result[:,0][np.asarray(result[:,1]).astype(int) == occurences]
        for eq in equations_to_remove:
            eq = eq.strip()
            removed = np.append(removed, [d for d in data if d['equation'].strip() == eq])
            data = [d for d in data if not d['equation'].strip() == eq]

        print('total # equations removed:', len(removed))
        occurences += 1
    return data, removed

def saveValues(indices, jsondata, output_path):
    output = open(output_path, 'w')
    for d in jsondata:
        if int(d['id']) in indices:
            output.write(str(d['variable_values']) + '\n')
    output.close()

def tsvs2tsv(common_path, uncommon_path, output_path):
    """
    takes tsv for both common and uncommon data
    writes a combined tsv with uncommon tgt replaced with 'seq'
    """
    common = open(common_path).readlines()
    uncommon = open(uncommon_path).readlines()
    output = open(output_path, 'w')
    for d in uncommon:
        result = d.split('\t')
        result[1] = 'seq\n'
        output.write('\t'.join(result))
    for d in common:
        output.write(d)
    output.close()

def splitTrainVal(train_val_path, output_train_path, output_val_path, num_val_examples=1000):
    """
    Sets aside num_val_examples [default: 1000] validation examples from
    train_val_path into output_val_path. All other examples are written to
    output_train_path
    train_val_path, output_train_path, output_val_path are tsv's
    """
    train_val = open(train_val_path).readlines()
    random.shuffle(train_val)
    output_train = open(output_train_path, 'w')
    output_val = open(output_val_path, 'w')
    for d in train_val[:num_val_examples]:
        output_val.write(d)
    for d in train_val[num_val_examples:]:
        output_train.write(d)
    output_train.close()
    output_val.close()

def preprocess(question, equation, sni_model, fields, use_sni=True):
    """
    Returns preprocessed version of question and equation using sni_model and
    fields
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
    #equation = equation.replace('%', ' / 100 ')
    print('equation before:', equation)
    equation = re.sub(r'(\d*.{0,1}\d+)%', r'(\1 / 100 )', equation)
    print('equation after:', equation)

    # preprocess question
    equation = equation.split()
    question = question.split()

    # prepend and postpend null tokens to question to allow for sni window size
    # of three
    question = ['null', 'null', 'null'] + question + ['null', 'null', 'null']

    # prevent inplace changes on question
    question_copy = [t for t in question]

    # replace significant numbers in question and equation
    i = 0
    variable_values = dict()
    for j,token in enumerate(question):
        if isFloat(token):
            example = question_copy[j-3:j+4]
            ex = data.Example.fromlist([' '.join(example), ''], fields)
            dataset = data.Dataset([ex], fields)
            inp = None
            iterator = data.Iterator(dataset, batch_size=1)
            iterator.repeat=False
            for batch in iterator:
                inp = batch.text.t()

            if (not use_sni) or (use_sni and isSignificant(inp, sni_model)):
                #if (use_sni and isSignificant(inp, sni_model)) or (not use_sni):
                for symbol in equation:
                    if symbol == token:
                        equation[equation.index(symbol)] = '[' + chr(97 + i) + ']'
                character = '[' +chr(97 + i) + ']'
                variable_values[character] = token
                for q in question:
                    if q == token:
                        question[question.index(q)] = '[' + chr(97 + i) + ']'
                i += 1

    # remove pre/postpended null tokens from question
    question = question[3:-3]

    question = ' '.join(question) + '\n'
    equation = ' '.join(equation) + '\n'
    return question, equation, variable_values

def json2tsv(json_indices, json_data, output_path):
    """
    For each example in json_data indexed by json_indices,
    writes the associated question and equation to output_path
    """
    output = open(output_path, 'w')
    for d in json_data:
        if int(d['id']) in json_indices:
            output.write(d['segmented_text'].strip() + '\t' +
                    d['equation'].strip() + '\t' + str(d['variable_values'])
                    + '\t' + str(d['ans']) + '\n')
    output.close()

def isFloat(value):
    """
    Returns True iff value can be represented as a float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False

def isSignificant(inp, sni_model):
    """
    Returns True iff inp is classified as significant by sni_model
    """
    return(evalTest.fast_test(inp, sni_model).data[0] == 1)

def txt2tsv(src_path, tgt_path, tsv_path):
    """
    Combines src_path and tgt_path into tsv_path
    """
    src_txt = open(src_path).readlines()
    tgt_txt = open(tgt_path).readlines()
    tsv = open(tsv_path, 'w')
    for i in range(len(src_txt)):
        tsv.write(src_txt[i].strip() + '\t' + tgt_txt[i].strip() +'\n')
    tsv.close()

def saveAnswers(json_indices, jsondata, output_path):
    output = open(output_path, 'w')
    parser = Parser()
    for d in jsondata:
        if int(d['id']) in json_indices:
            #print(str(d['ans']) + '\n')
            answer = answer.replace('%', ' / 100')
            try:
                answer = parser.evaluate(answer, variables=None)
            except:
                pass
            output.write(str(d['ans']) + '\n')
    output.close()

if __name__ == '__main__':
    main()
