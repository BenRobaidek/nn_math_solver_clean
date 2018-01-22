import json
import numpy as np
from py_expression_eval import Parser

def main():
    print('Running main...')
    jsondata = json.loads(open('../tencent/data/input/Math23K.json').read())

    for example in jsondata:
        solveEquation(exampe['equation'])

def solveEquation(equation):
    print('Solving Equation:', equation)

if __name__ == '__main__':
    main()
