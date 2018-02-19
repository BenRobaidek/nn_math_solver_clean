import solver

pred_equations = open('./data/basic/').readlines()
tgt_equations =
variables =
answers =

corrects = solver.solve(tgt_equations, variables, answers)

print('tgts correspond to answer:', 100*np.sum(corrects)/len(corrects))
