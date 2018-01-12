import argparse
import numpy as np

def main(args):
    """
    Print accuracy of guessing correct equation
    """
    val_tgts = open(args.inp).readlines()
    val_tgts = [line.split('\t')[1].strip() for line in val_tgts]
    #val_tgts = np.expand_dims(val_tgts, axis=1)
    preds = open(args.preds).readlines()
    preds = [line.strip() for line in preds]
    #preds = np.expand_dims(preds, axis=1)
    correct = 0
    total = len(preds)
    for i,ex in enumerate(preds):
        if preds[i] == val_tgts[i]:
            correct += 1
    print('Accuracy of guessing equation:', correct/total, '(', correct, '/', total, ')')
    #for i,line in enumerate(val_tgts):
    #    if not val_tgts[i] == preds[i]:
    #        #print(val_tgts[i], preds[i])
    #        #print(i)


def parseArgs():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-inp', type=str, default='../tencent/data/working/basic/val.tsv', help='path to inp file')
    parser.add_argument('-preds', type=str, default='../tests/train_best_models/saved_models/classifier_basic/preds.txt', help='path to preds file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
