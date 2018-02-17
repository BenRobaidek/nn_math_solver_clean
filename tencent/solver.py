"""
Solver for tencent problems
"""
import numpy as np
import re

def solve(equations, variables, answers):
    corrects = np.array([])
    for eq, var, ans in zip(equations, variables, answers):
        # eval on var dict str
        var = eval(var)

        # sub variables into predicted and target equations
        for k in var:
            eq = eq.replace(k, var[k])

        # Add multiplication symbols to answer where needed
        ans = re.sub(r'\(\((\d+)\)/\((\d+)\)\)',r'(\1/\2)',ans)
        ans = re.sub(r'(\d)\(',r'\1+(', ans, 1)
        # replace % in answer
        ans = ans.replace('%', ' * .01')
        ans = eval(ans)

        # replace ^ with ** in predicted equation
        eq = eq.replace('^', '**')

        # remove = from equations
        eq = eq.strip('x =')

        # evaluate
        eq = eq.strip()

        examples_not_working = 0
        if (not eq == '80千米 / 小时') and (not re.search(r'\[\S\]', eq)) and (not eq == '<unk>'):
            try:
                eq = eval(eq)
            except (ZeroDivisionError, OverflowError, SyntaxError):
                examples_not_working += 1
                pass

        try:
            if abs(float(eq) - float(ans)) < .002:
                corrects = np.append(corrects, [True])
            else:
                corrects = np.append(corrects, [False])
        except (ValueError, OverflowError) as e:
            """
            if calculating tgt eq to answer correlation, the line belod should
            be True as this corresponds to the SNI messing up
            """
            corrects = np.append(corrects, [False])
            examples_not_working += 1

    print('examples_not_working', examples_not_working)
    return corrects.astype(bool)
