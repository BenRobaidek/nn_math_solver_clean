import json
import numpy as np
import json
import copy
import re
import math

def main():
    # LOAD JSON DATA
    jsondata = json.loads(open('./input/draw.json').read())
    jsondata_no_sni = copy.deepcopy(jsondata)

    # PREPROCESS DATA WITHOUT SNI
    #TODO

    # PREPROCESS DATA WITHOUT SNI
    print('Preprocessing without sni...')
    for d in jsondata_no_sni:
        d['sQuestion'], d['Template'] = preprocess(d['sQuestion'], d['Template'], d['Alignment'], model=None, fields=None, use_sni=False)
    print('Preprocessing without sni complete...')

    # LOAD INDICES
    train_indicesk1234 = [int(x.strip()) for x in open('./input/traink.txt').readlines()]
    val_indicesk1234 = [int(x.strip()) for x in open('./input/val.txt').readlines()]
    test_indicesk5 = [int(x.strip()) for x in open('./input/test.txt').readlines()]

    # OUTPUT TO JSON FILES
    jsonToTsv(train_indices,  jsondata_no_sni, './working/no_sni/traink1234.tsv')
    jsonToTsv(val_indices,    jsondata_no_sni, './working/no_sni/valk1234.tsv')
    jsonToTsv(test_indices,   jsondata_no_sni, './working/no_sni/testk5.tsv')

def preprocess(question, template, alignment, model, fields, use_sni=True):
    # Preprocess Question
    question = question.strip() + ' '
    if re.search('( [.?] )$', question) is None: question = question + ' . '
    if re.search('( [.?] )', question) is not None: question = re.split('( [.?] )', question)
    else: question = re.split('( [.?] )', question + ' . ')
    #if not question[-1] == '.': question = question + ['.']
    question = [x.split() for x in question]
    question = question[0:-1]
    question = [np.append(question[2*i], question[2*i+1]).astype(str) for i in range(0,int(math.floor(len(question))/2))]

    for constant in alignment:
        question[constant['SentenceId']][constant['TokenId']] = '[' + constant['coeff'] + ']'

    question = [' '.join(x) for x in question]
    question = ' '.join(question)

    # Preprocess Equations
    for j,eq in enumerate(template):
        symbols = eq.split()
        for i,symbol in enumerate(symbols):
            if symbol not in ['+', '-', '*', '/', '(', ')', '='] and not isFloat(symbol):
                symbols[i] = '[' + symbol + ']'
        template[j] = ' '.join(symbols)
    print('template:', template)

    return question, template


def jsonToTsv(indices, data, output_path):
    output = open(output_path, 'w')
    for d in data:
        if d['iIndex'] in indices:
            result = d['sQuestion'] + '\t'
            result = result + str(','.join(d['Template'])) + '\t'
            result = result + str(d['Alignment']) + '\t'
            result = result + str(d['lSolutions']) + '\n'
            output.write(result)
    output.close()

def isFloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

if __name__ == '__main__':
    main()
