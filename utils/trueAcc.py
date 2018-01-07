import operator
import argparse
import numpy as np

def main(args):
    """
    Print true accuracy
    """
    # LOAD DATA FILES
    variables = open(args.variables).readlines()
    preds = open(args.preds).readlines()
    answers = open(args.answers, 'w')

    # COMPUTE ANSWERS


    # CLOSE FILES
    answers.close()

def parseArgs():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-variables', type=str, default='../tencent/data/working/basic/val_values.txt', help='path to variable values [default: \'../tencent/data/working/basic/val_values.txt\']')
    parser.add_argument('-preds', type=str, default='../tencent/data/output/basic/preds.txt', help='path to preds [default: \'../tencent/data/working/preds.txt\']')
    parser.add_argument('-answers', type=str, default='../tencent/data/output/basic/answers.txt', help='path to save answers [default: \'../tencent/data/output/basic/answers.txt\']')
     
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    main(args)
