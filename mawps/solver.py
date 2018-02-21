"""
Solver for mawps problems
"""
import numpy as np
import re

def solve(equations, variables, answers):
    corrects = np.array([])
    test_dicts = open('../mawps/data/test.dicts').readlines()
    test = [x.split('\t')[1] for x in open('../mawps/data/test.tsv').readlines()]
    for pred_eq, gold_eq, var in zip(equations, test, test_dicts):
        var = eval(var)
        print('pred_eq', pred_eq)
        print('gold_eq', gold_eq)
        print('var', var)


    #print('test_dicts:', test_dicts)
    corrects = np.ones(len(equations))
    return corrects.astype(bool)
