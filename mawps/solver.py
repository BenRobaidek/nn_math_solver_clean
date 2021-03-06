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
    test = [x.split('\t')[1] for x in open('../mawps/data/working/basic/testk5.tsv').readlines()]
    #print(len(equations), len(test), len(test_dicts))
    for pred_eq, gold_eq, var in zip(equations, test, test_dicts):
        var = eval(var)
        #print('pred_eq', pred_eq)
        #print('gold_eq', gold_eq)
        #print('var', var)

        for k in var.keys():
            pred_eq = pred_eq.replace(var.get(k), str(k))
            gold_eq = gold_eq.replace(var.get(k), str(k))

        pred_answer = None
        if (not pred_eq.strip()=='<unk>') and ('[' not in pred_eq):
            pred_eq = '(' + pred_eq.split('=')[1] + ') - (' + pred_eq.split('=')[0] + ')'
            expr = parse_expr(pred_eq)
            symbols = sympy.symbols('x')
            pred_answer = sympy.solve(expr, symbols)
            if pred_answer == []:
                pred_answer = None
            else:
                pred_answer = pred_answer[0]

        gold_answer = None
        if (not gold_eq.strip()=='<unk>') and ('[' not in gold_eq):
            gold_eq = '(' + gold_eq.split('=')[1] + ') - (' + gold_eq.split('=')[0] + ')'
            expr = parse_expr(gold_eq)
            symbols = sympy.symbols('x')
            gold_answer = sympy.solve(expr, symbols)
            if gold_answer == []:
                gold_answer = None
            else:
                gold_answer = gold_answer[0]

        if pred_answer is not None and gold_answer is not None and abs(pred_answer - gold_answer) < .002:
            corrects = np.append(corrects, [True])
        else:
            corrects = np.append(corrects, [False])

    #print('test_dicts:', test_dicts)
    return corrects.astype(bool)
