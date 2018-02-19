"""
Solver for ms_draw problems
"""
import numpy as np
import re
import sympy
from sympy.parsing.sympy_parser import parse_expr

def solve(equations, variables, answers):
    corrects = np.array([])
    for eq, var, ans in zip(equations, variables, answers):
        #print('eq:', eq)
        #print('var:', var)
        #print('ans:', ans)asdfasdf
        ans = eval(ans)
        for i,a in enumerate(ans):
            #print('type(a):', type(a))
            ans[i] = float(a)

        # sub variables into predicted and target equations
        var = eval(var)
        for constant in var:
            eq = eq.replace('[' + constant['coeff'] + ']', str(constant['Value']))

        # replace ^ with ** in predicted equation
        eq = eq.replace('^', '**')

        #print('eq:', eq)
        #print('var:', var)
        #print('ans:', ans)

        print('eq:', eq)
        if (not eq.strip() == '<unk>'):

            # get variables out of predicted equation
            answer_variables = np.unique(re.findall(r'\[[mnop]\]', eq, flags=0))

            eq = eq.split(',')
            for k,p in enumerate(eq):
                eq[k] = '(' + p.split('=')[1] + ') - (' + p.split('=')[0] + ')'

            #if len(eq) == 1: eq = eq[0]
            pred_answers = dict()
            #print(answer_variables)

            #print('answer_variables:', answer_variables)
            if not len(np.unique(re.findall(r'\[[a-l]\]', ','.join(eq)))) >= 1:
                #print('eq:', eq)
                expr = [parse_expr(x.replace('[', '').replace(']', '')) for x in eq]
                print(answer_variables)
                symbols = sympy.symbols(' '.join(answer_variables))
                pred_answers = sympy.solve(expr)#, symbols)
                #print('pred_answers:', pred_answers)

            all_equal = False
            if len(pred_answers) == len(ans):
                #print(pred_answers)
                #print('pred_answers:', pred_answers)
                #print('answer:', answer)
                differences = np.absolute(np.subtract(np.array(list(pred_answers.values())).astype(float), ans))
                #print('differences:', differences)
                correct_answers = np.less(differences, np.ones(np.shape(pred_answers.values())) * .002)
                #print('correct_answers:', correct_answers)
                all_equal = np.all(correct_answers)
                #print('all_equal:', all_equal)
                #print()
            corrects = np.append(corrects, [all_equal])
        elif (eq.strip() == '<unk>'):
            corrects = np.append(corrects, [False])

    return corrects.astype(bool)
