import argparse
import numpy as np
from torchtext import data

def main(args):
    print('Running main...')
    args.classifier_answers = './classifier_answers.txt'
    args.seq2seq_answers = './seq2seq_answers.txt'
    normalAcc(args)

def normalAcc(args):
    classifer_answers = open(arg.classifier_answers).readlines()
    seq2seq_answers = open(arg.seq2seq_answers).readlines()

def parseArgs():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-classifier_answers', type=str, default='../tencent/data/working/basic/train.tsv', help='path to train tsv file [default: \'../tencent/data/working/basic/train.tsv\']')
    parser.add_argument('-seq2seq_answers', type=str, default='../tencent/data/working/basic/val.tsv', help='path to input tsv file, usually validation file [default: \'../tencent/data/working/basic/val.tsv\']')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    print('args:', args)
    main(args)
