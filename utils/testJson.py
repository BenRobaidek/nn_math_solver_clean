import json
import numpy as np
from ..tencent.data.preprocess import preprocess
from py_expression_eval import Parser

def main():
    print('Running main...')
    jsondata = json.loads(open('../tencent/data/input/Math23K.json').read())

    for d in jsondata:
        d['segmented_text'], d['equation'], d['variable_values'] = preprocess(d['segmented_text'], d['equation'], model, fields, use_sni=True)
        solveEquation(d)

def solveEquation(d):
    print('Solving Equation:', d['equation'])

if __name__ == '__main__':
    main()
