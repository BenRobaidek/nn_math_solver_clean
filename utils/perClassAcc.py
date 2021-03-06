import operator
import argparse
import numpy as np
import torch

def main(args):
    """
    Compute/print per class accuracy
    """
    itos = torch.load('../classifier/LABELS_vocab_itos.pt')
    predictions = open('../classifier/predictions.txt').readlines()
    perClassAcc_file = open('../classifier/perClassAcc.txt', 'w')

    #for eq in itos:
    #    print(eq)

    results = np.ones([len(itos),3])
    results = -1 * results

    for line in predictions:
        equation, prediction, target = line.split('\t')


        if results[itos.index(equation),0] == -1 or results[itos.index(equation),1] == -1:
            results[itos.index(equation),0] = 0
            results[itos.index(equation),1] = 0
        if isFloat(prediction): prediction = float(prediction)
        if isFloat(target): target = float(target)
        if isFloat(prediction) and isFloat(target) and abs(prediction - target) <= .002:
            results[itos.index(equation),0] += 1
        else:
            results[itos.index(equation),1] += 1

    #print('len(itos):', np.unique(len(itos)))

    all_results = []
    for eq, right, wrong, acc in zip(itos, results[:,0], results[:,1], results[:,2]):
        all_results = np.append(all_results, [eq, right, wrong, acc], axis=0)
    all_results = all_results.reshape(len(itos), -1)
    for line in all_results:
        line[1] = float(line[1])
        line[2] = float(line[2])
        eq, right, wrong = line[0], float(line[1]), float(line[2])
        if right == -1 or wrong == -1:
            right == None
            wrong == None
            line[3] = 'NA'
        else:
            line[3] = right/(right+wrong)

    all_results = sorted(all_results, key=lambda x: x[3], reverse=True)
    for line in all_results:
        perClassAcc_file.write('' + line[0] + ' true_acc: ' + line[3] + ' ({}/{})'.format(line[1],float(line[1])+float(line[2])) + '\n')
    perClassAcc_file.close()

    all_results = np.array(all_results)
    #print('Val Acc:', np.sum(all_results[:,1]))

def isFloat(f):
    try:
        float(f)
        return True
    except Exception as e:
        False

def parseArgs():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-truth', type=str, default='../tencent/data/working/basic/val.tsv', help='path to ground true file, usually validation file [default: '']')
    parser.add_argument('-preds', type=str, default='./preds.txt', help='path to preds file [default: '']')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
