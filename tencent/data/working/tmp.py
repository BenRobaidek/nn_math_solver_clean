def main():
    split('./basic/train.tsv', './s2s/src-train.txt', './s2s/tgt-train.txt')
    split('./basic/val.tsv', './s2s/src-val.txt', './s2s/tgt-val.txt')

    split('./basic/traink123.tsv', './s2s/src-traink123.txt', './s2s/tgt-traink123.txt')
    split('./basic/valk4.tsv', './s2s/src-valk4.txt', './s2s/tgt-valk4.txt')
    split('./basic/testk5.tsv', './s2s/src-testk5.txt', './s2s/tgt-testk5.txt')

    split('./basic/traink234.tsv', './s2s/src-traink234.txt', './s2s/tgt-traink234.txt')
    split('./basic/valk5.tsv', './s2s/src-valk5.txt', './s2s/tgt-valk5.txt')
    split('./basic/testk1.tsv', './s2s/src-testk1.txt', './s2s/tgt-testk1.txt')

    split('./basic/traink345.tsv', './s2s/src-traink345.txt', './s2s/tgt-traink345.txt')
    split('./basic/valk1.tsv', './s2s/src-valk1.txt', './s2s/tgt-valk1.txt')
    split('./basic/testk2.tsv', './s2s/src-testk2.txt', './s2s/tgt-testk2.txt')

    split('./basic/traink451.tsv', './s2s/src-traink451.txt', './s2s/tgt-traink451.txt')
    split('./basic/valk2.tsv', './s2s/src-valk2.txt', './s2s/tgt-valk2.txt')
    split('./basic/testk3.tsv', './s2s/src-testk3.txt', './s2s/tgt-testk3.txt')

    split('./basic/traink512.tsv', './s2s/src-traink512.txt', './s2s/tgt-traink512.txt')
    split('./basic/valk3.tsv', './s2s/src-valk3.txt', './s2s/tgt-valk3.txt')
    split('./basic/testk4.tsv', './s2s/src-testk4.txt', './s2s/tgt-testk4.txt')

def split(tsv, src_txt, tgt_txt):
    tsv = open(tsv).readlines()
    with open(src_txt, 'w') as src_txt:
        for x in tsv:
            src_txt.write(x.split('\t')[0] + '\n')
    with open(tgt_txt, 'w') as tgt_txt:
        for x in tsv:
            tgt_txt.write(x.split('\t')[1] + '\n')

if __name__ == '__main__':
    main()
