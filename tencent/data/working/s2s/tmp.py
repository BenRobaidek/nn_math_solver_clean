def main():
    printClassAcc('predk1.txt', 'tgt-valk1.txt')
    printClassAcc('predk2.txt', 'tgt-valk2.txt')
    printClassAcc('predk3.txt', 'tgt-valk3.txt')
    printClassAcc('predk4.txt', 'tgt-valk4.txt')
    printClassAcc('predk5.txt', 'tgt-valk5.txt')

def printClassAcc(preds_path, tgts_path):
    preds = open(preds_path).readlines()
    tgts = open(tgts_path).readlines()
    corrects = 0
    for pred, tgt in zip(preds, tgts):

        if pred.strip() == tgt.strip():
            corrects += 1
        #else:
        #    print('pred:', pred.strip(), 'tgt:', tgt.strip())

    print('class accuracy:', corrects/len(tgts))

if __name__ == '__main__':
    main()
