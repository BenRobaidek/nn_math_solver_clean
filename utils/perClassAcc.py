def main(args):
    """
    Compute/print per class accuracy
    """
    truth = open(args.truth).readlines()
    preds = open(args.preds).readlines()
    print(truth)
    print(preds)


def parseArgs():
    parser = argparse.ArgumentParser(description='test')

    parser.add_argument('-truth', type=str, default='./preds.txt', help='path to ground true file, usually validation file [default: '']')
    parser.add_argument('-preds', type=str, default='./truth.txt', help='path to preds file [default: '']')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
