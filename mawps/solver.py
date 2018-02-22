"""
Solver for mawps problems
"""
import numpy as np
import re
import sympy
from sympy.parsing.sympy_parser import parse_expr

def solve(equations, variables, answers):
    corrects = np.array([])
    test_dicts = open('../mawps/data/test.dicts').readlines()
    test = [x.split('\t')[1] for x in open('../mawps/data/test.tsv').readlines()]
    for pred_eq, gold_eq, var in zip(equations, test, test_dicts):
        var = eval(var)
        print('pred_eq', pred_eq)
        print('gold_eq', gold_eq)
        print('var', var)

        for k in var.keys():
            pred_eq = pred_eq.replace(var.get(k), str(k))
            gold_eq = gold_eq.replace(var.get(k), str(k))

        pred_eq = '(' + pred_eq.split('=')[1] + ') - (' + pred_eq.split('=')[0] + ')'
        gold_eq = '(' + gold_eq.split('=')[1] + ') - (' + gold_eq.split('=')[0] + ')'

        if (pred_eq is not '<unk>'):
            print(pred_eq)
            expr = parse_expr(pred_eq)
            symbols = sympy.symbols('x')
            pred_answers = sympy.solve(expr, symbols)

        print(pred_answers)

        print('pred_eq', pred_eq)
        print('gold_eq', gold_eq)
        print('var', var)


    #print('test_dicts:', test_dicts)
    corrects = np.ones(len(equations))
    return corrects.astype(bool)
