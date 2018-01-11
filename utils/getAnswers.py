import operator
import argparse
import numpy as np
import ast
from py_expression_eval import Parser
import re

def main(args):
    """
    Print true accuracy
    """
    equations = [line for line in open(args.equations).readlines()]
    variables = [ast.literal_eval(line) for line in open(args.variables).readlines()]

    answers = open(args.answers, 'w')
    for eq, var in zip(equations, variables):
        answers.write(str(solve(eq, var)))
    answers.close()

def solve(equation, variables):
    """
    Given an equation and variables, solves the equation if possible.
    If not possible to solve equation, returns 'no answer'
    """
    for key in variables.keys():
        equation = equation.replace(key, variables[key])
    equation = equation.strip('x =')
    print('equation:', equation)
    print('variables:', variables)
    parser = Parser()
    print(re.search('[[a-z]]', equation))
    if re.search('[[a-z]]', equation) is not None:
        answer = 'no answer'
    else:
        answer = parser.evaluate(equation, variables=None)
    return answer


def parseArgs():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('-equations', type=str, default='../tencent/data/output/basic/preds.txt', help='path to equations file, usually preds.txt [default: '']')
    parser.add_argument('-variables', type=str, default='../tencent/data/working/basic/val_values.txt', help='path to validation variable values file [default: '']')
    parser.add_argument('-answers', type=str, default='../tencent/data/output/basic/answers.txt', help='path to save answers [default: '']')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
