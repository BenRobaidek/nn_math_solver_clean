"""
Solver for kushman problems
"""
import numpy as np
import re
import sympy
from sympy.parsing.sympy_parser import parse_expr

def solve(equations, variables, answers):
    corrects = np.array([])
    for eq, var, ans in zip(equations, variables, answers):
        ans = eval(ans)
        for i,a in enumerate(ans):
            #print('type(a):', type(a))
            ans[i] = float(a)

        # sub variables into predicted and target equations
        var = eval(var)
        for k in var:
            eq = eq.replace(k, str(var[k]))

        # replace ^ with ** in predicted equation
        eq = eq.replace('^', '**')

        if (eq is not '<unk>') and '=' in eq:

            # get variables out of predicted equation
            answer_variables = np.unique(re.findall(r'VAR_[\d]', eq, flags=0))

            eq = eq.split(',')
            for k,p in enumerate(eq):
                eq[k] = '(' + p.split('=')[1] + ') - (' + p.split('=')[0] + ')'

            #if len(eq) == 1: eq = eq[0]
            pred_answers = dict()
            #print(answer_variables)

            if not len(np.unique(re.findall(r'\[[a-z]\]', ','.join(eq)))) >= 1:
                #print('eq:', eq)
                expr = [parse_expr(x) for x in eq]
                symbols = sympy.symbols(' '.join(answer_variables))
                pred_answers = sympy.solve(expr)#, symbols)

            all_equal = False
            if len(pred_answers) == len(ans):
                #print(pred_answers)
                #print('pred_answers:', pred_answers)
                #print('answer:', answer)
                differences = np.absolute(np.subtract(np.array(list(pred_answers.values())).astype(float), answer))
                #print('differences:', differences)
                correct_answers = np.less(differences, np.ones(np.shape(pred_answers.values())) * .002)
                #print('correct_answers:', correct_answers)
                all_equal = np.all(correct_answers)
                #print('all_equal:', all_equal)
                #print()
            corrects = np.append(corrects, [all_equal])
    return corrects.astype(bool)
