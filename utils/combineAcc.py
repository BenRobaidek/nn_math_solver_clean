import operator
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

def main(args):
    """
    prints combined classifier and seq2seq true accuracy
    """
    s2s_preds = open(args.s2s_preds).readlines()
    s2s_preds = [line.strip() == 'True' for line in s2s_preds]

    classifier_preds = np.array(open(args.classifier_preds).readlines())
    classifier_preds = np.array([line.split(' ') for line in classifier_preds])
    classifier_probabilities = np.array(classifier_preds[:,1])
    classifier_probabilities = np.array([line.strip() for line in classifier_probabilities]).astype(float)
    classifier_preds = [line == 'True' for line in classifier_preds[:,0]]
    x = np.divide(list(range(0,100)), 100)
    print(x)
    y = [getCombinedAcc(classifier_probabilities, classifier_preds, s2s_preds, i) for i in x]
    print('classifier acc:', np.sum(classifier_preds)/len(classifier_preds))
    print('s2s acc:', np.sum(s2s_preds)/len(s2s_preds))
    plt.plot(x, y)
    plt.ylabel('combined true acc')
    plt.xlabel('probability threshold')
    plt.show()

def getCombinedAcc(classifier_probabilities, classifier_preds, s2s_preds, threshold):
    correct = 0
    for probability, classifier_pred, s2s_pred in zip(classifier_probabilities, classifier_preds, s2s_preds):
        if probability > threshold:
            correct += int(classifier_pred)
        else:
            correct += int(s2s_pred)

    #print('combined acc:', correct/len(classifier_preds))
    return(correct/len(classifier_preds))

def parseArgs():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-s2s_preds', type=str, default='/tmp/predictions.txt.correct', help='path to ground s2s predictions [default: '']')
    parser.add_argument('-classifier_preds', type=str, default='../classifier/predictions_62.4true.txt', help='path to classifier predictions file [default: '']')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
