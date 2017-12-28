import json
import numpy as np
import random
import math
import re
import sys
import torch
from torchtext import data, datasets

sys.path.append('../../sni/model')
import model
import evalTest

def main():

    # LOAD JSON DATA
    jsondata = json.loads(open('./Math23K.json').read())

    # LOAD SNI MODEL
    model = torch.load('../../sni/models/sni_best_model.pt')
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
    train_classifier = data.TabularDataset(path='./train.tsv', format='tsv', fields=fields)
    LABEL.build_vocab(train)

    # PREPROCESS DATA
    print('Preprocessing...')
    for d in jsondata:
        d['segmented_text'], d['equation'] = preprocess(d['segmented_text'], d['equation'], model, fields)
    print('Preprocessing Complete...')

    with open('./Math23K-preprocessed.json', 'w') as outfile:
        json.dump(jsondata, outfile)

    # 5 FOLD CROSS VALIDATION
    print('Using existing cross validation splits')
    #print('Preforming cross validation splits...')
    #crossValidation(jsondata, k = 5, k_test=5)

    # SAVE SPLIT INDICES
    split('./Math23K-train.txt', './Math23K-dev.txt', './Math23K-test.txt', k_test=5)

    # SAVE SRC/TGT files
    train_indices = np.genfromtxt('./Math23K-train.txt').astype(int)
    dev_indices = np.genfromtxt('./Math23K-dev.txt').astype(int)
    test_indices = np.genfromtxt('./Math23K-test.txt').astype(int)
    json2txt(train_indices, jsondata,   './src-train.txt',  './tgt-train.txt')
    json2txt(dev_indices,   jsondata,   './src-val.txt',    './tgt-val.txt')
    json2txt(test_indices,  jsondata,   './src-test.txt',   './tgt-test.txt')

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

    # SAVE SRC/TGT FILES (FILTERED DATA)
    train_dev_indices = np.append(train_indices, dev_indices)
    json2txt(train_dev_indices, common_data2,    './src-train_dev_0.2_common.txt',   './tgt-train_dev_0.2_common.txt')
    json2txt(train_dev_indices, uncommon_data2,  './src-train_dev_0.2_uncommon.txt', './tgt-train_dev_0.2_uncommon.txt')

    json2txt(train_dev_indices, common_data4,    './src-train_dev_0.4_common.txt',   './tgt-train_dev_0.4_common.txt')
    json2txt(train_dev_indices, uncommon_data4,  './src-train_dev_0.4_uncommon.txt', './tgt-train_dev_0.4_uncommon.txt')

    json2txt(train_dev_indices, common_data6,    './src-train_dev_0.6_common.txt',   './tgt-train_dev_0.6_common.txt')
    json2txt(train_dev_indices, uncommon_data6,  './src-train_dev_0.6_uncommon.txt', './tgt-train_dev_0.6_uncommon.txt')

    json2txt(train_dev_indices, common_data8,    './src-train_dev_0.8_common.txt',   './tgt-train_dev_0.8_common.txt')
    json2txt(train_dev_indices, uncommon_data8,  './src-train_dev_0.8_uncommon.txt', './tgt-train_dev_0.8_uncommon.txt')

    # SAVE TSV FILES
    txt2tsv('./src-train.txt',  './tgt-train.txt', './train.tsv')
    txt2tsv('./src-val.txt',  './tgt-val.txt', './val.tsv')
    txt2tsv('./src-test.txt',  './tgt-test.txt', './test.tsv')
    txt2tsv('./src-train_dev_0.2_common.txt',   './tgt-train_dev_0.2_common.txt',   './train_dev_0.2_common.tsv')
    txt2tsv('./src-train_dev_0.2_uncommon.txt', './tgt-train_dev_0.2_uncommon.txt', './train_dev_0.2_uncommon.tsv')
    txt2tsv('./src-train_dev_0.4_common.txt',   './tgt-train_dev_0.4_common.txt',   './train_dev_0.4_common.tsv')
    txt2tsv('./src-train_dev_0.4_uncommon.txt', './tgt-train_dev_0.4_uncommon.txt', './train_dev_0.4_uncommon.tsv')
    txt2tsv('./src-train_dev_0.6_common.txt',   './tgt-train_dev_0.6_common.txt',   './train_dev_0.6_common.tsv')
    txt2tsv('./src-train_dev_0.6_uncommon.txt', './tgt-train_dev_0.6_uncommon.txt', './train_dev_0.6_uncommon.tsv')
    txt2tsv('./src-train_dev_0.8_common.txt',   './tgt-train_dev_0.8_common.txt',   './train_dev_0.8_common.tsv')
    txt2tsv('./src-train_dev_0.8_uncommon.txt', './tgt-train_dev_0.8_uncommon.txt', './train_dev_0.8_uncommon.tsv')

    # SAVE FULL TSV FILES
    tsvs2tsv('./train_dev_0.2_common.tsv', './train_dev_0.2_uncommon.tsv', './train_dev_0.2.tsv')
    tsvs2tsv('./train_dev_0.4_common.tsv', './train_dev_0.4_uncommon.tsv', './train_dev_0.4.tsv')
    tsvs2tsv('./train_dev_0.6_common.tsv', './train_dev_0.6_uncommon.tsv', './train_dev_0.6.tsv')
    tsvs2tsv('./train_dev_0.8_common.tsv', './train_dev_0.8_uncommon.tsv', './train_dev_0.8.tsv')

    # SAVE FULL TXT FILES FOR SEQ2SEQ
    tsvs2txt('./train_dev_0.2_common.tsv', './train_dev_0.2_uncommon.tsv', './src-train_dev_0.2.txt', './tgt-train_dev_0.2.txt')
    tsvs2txt('./train_dev_0.4_common.tsv', './train_dev_0.4_uncommon.tsv', './src-train_dev_0.4.txt', './tgt-train_dev_0.4.txt')
    tsvs2txt('./train_dev_0.6_common.tsv', './train_dev_0.6_uncommon.tsv', './src-train_dev_0.6.txt', './tgt-train_dev_0.6.txt')
    tsvs2txt('./train_dev_0.8_common.tsv', './train_dev_0.8_uncommon.tsv', './src-train_dev_0.8.txt', './tgt-train_dev_0.8.txt')

    # SPLIT TRAIN DEV FOR CLASSIFIER
    splitTrainVal('./train_dev_0.2.tsv', './train_0.2.tsv', './dev_0.2.tsv')
    splitTrainVal('./train_dev_0.4.tsv', './train_0.4.tsv', './dev_0.4.tsv')
    splitTrainVal('./train_dev_0.6.tsv', './train_0.6.tsv', './dev_0.6.tsv')
    splitTrainVal('./train_dev_0.8.tsv', './train_0.8.tsv', './dev_0.8.tsv')

    # SPLIT TRAIN DEV FOR SEQ2SEQ
    splitTrainVal('./src-train_dev_0.2.txt', './src-train_0.2.txt', './src-dev_0.2.txt')
    splitTrainVal('./tgt-train_dev_0.2.txt', './tgt-train_0.2.txt', './tgt-dev_0.2.txt')
    splitTrainVal('./src-train_dev_0.4.txt', './src-train_0.4.txt', './src-dev_0.4.txt')
    splitTrainVal('./tgt-train_dev_0.4.txt', './tgt-train_0.4.txt', './tgt-dev_0.4.txt')
    splitTrainVal('./src-train_dev_0.6.txt', './src-train_0.6.txt', './src-dev_0.6.txt')
    splitTrainVal('./tgt-train_dev_0.6.txt', './tgt-train_0.6.txt', './tgt-dev_0.6.txt')
    splitTrainVal('./src-train_dev_0.8.txt', './src-train_0.8.txt', './src-dev_0.8.txt')
    splitTrainVal('./tgt-train_dev_0.8.txt', './tgt-train_0.8.txt', './tgt-dev_0.8.txt')

    # SAVE VOCAB
    TEXT_class = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABEL_class = data.Field(sequential=False)
    fields_class = [('text', TEXT_class), ('label', LABEL_class)]
    train_class = data.TabularDataset(path='./train.tsv', format='tsv', fields=fields_class)
    TEXT_class.build_vocab(train_class)
    LABEL_class.build_vocab(train_class)
    torch.save((list(TEXT_class.vocab.stoi), list(LABEL_class.vocab.stoi)), 'vocab.pt')

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

def split(train_path, val_path, test_path, k_test=5):
    """
    Writes train, validation, and test indices to train_path, val_path, and
    test_path respectively
    foldi.txt files must already exist
    """
    train_val = []
    for i in range(1,6):
        if not i == k_test:
            train_val = np.append(train_val, open('fold' + str(i) + '.txt').readlines())
    #random.shuffle(train_val)
    test = open('fold' + str(k_test) + '.txt').readlines()

    # Train
    output = open(train_path, 'w')
    for d in train_val[0:-1000]:
        output.write(d)
    output.close()
    print(train_path + ' saved')

    # Validation
    output = open(val_path, 'w')
    for d in train_val[-1000:]:
        output.write(d)
    output.close()
    print(val_path + ' saved')

    # Test
    output = open(test_path, 'w')
    for d in test:
        output.write(d)
    output.close()
    print(test_path + ' saved')

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

def tsvs2txt(common_path, uncommon_path, output_path_src, output_path_tgt):
    """
    Takes tsv's common_path and uncommon_path
    writes combined txt's with uncommon tgt replaced with 'seq'
    """
    common = open(common_path).readlines()
    uncommon = open(uncommon_path).readlines()
    output_src = open(output_path_src, 'w')
    output_tgt = open(output_path_tgt, 'w')
    for d in uncommon:
        result = d.split('\t')
        result[0] = result[0].strip() + '\n'
        result[1] = 'seq\n'
        output_src.write(result[0])
        output_tgt.write(result[1])
    for d in common:
        result = d.split('\t')
        result[0] = result[0].strip() + '\n'
        result[1] = result[1].strip() + '\n'
        output_src.write(result[0])
        output_tgt.write(result[1])
    output_src.close()
    output_tgt.close()

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

def preprocess(question, equation, sni_model, fields):
    """
    Returns preprocessed version of question and equation using sni_model and
    fields
    """
    # handle %'s
    question = question.replace('%', ' % ')

    # handle fractions
    fractions = re.findall('\(\d+\)/\(\d+\)', question)
    fractions = np.append(fractions, re.findall('\(\d+/\d+\)', question))
    for i,fraction in enumerate(fractions):
        question = question.replace(fraction, str(sys.maxsize - i))
        equation = equation.replace(fraction, str(sys.maxsize - i))

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
            if isSignificant(inp, sni_model):
                for symbol in equation:
                    if symbol == token:
                        equation[equation.index(symbol)] = '[' + chr(97 + i) + ']'
                for q in question:
                    if q == token:
                        question[question.index(q)] = '[' + chr(97 + i) + ']'
                i += 1

    # remove pre/postpended null tokens from question
    question = question[3:-3]

    question = ' '.join(question) + '\n'
    equation = ' '.join(equation) + '\n'
    return question, equation

def json2txt(json_indices, json_data, output_path_src, output_path_tgt):
    """
    For each example in json_data indexed by json_indices,
    writes the associated question and equation to output_path_src
    and output_path_tgt respectively
    """
    output_src = open(output_path_src, 'w')
    output_tgt = open(output_path_tgt, 'w')
    for d in json_data:
        if int(d['id']) in json_indices:
            question, equation = d['segmented_text'], d['equation']
            output_src.write(question)
            output_tgt.write(equation)
    output_src.close()
    output_tgt.close()

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
    src_txt.close()
    tgt_txt.close()
    tsv.close()

if __name__ == '__main__':
    main()
