def main():
    split('./basic/traink1.tsv', './s2s/src-traink1.txt', './s2s/tgt-traink1.txt')
    split('./basic/traink2.tsv', './s2s/src-traink2.txt', './s2s/tgt-traink2.txt')
    split('./basic/traink3.tsv', './s2s/src-traink3.txt', './s2s/tgt-traink3.txt')
    split('./basic/traink4.tsv', './s2s/src-traink4.txt', './s2s/tgt-traink4.txt')
    split('./basic/traink5.tsv', './s2s/src-traink5.txt', './s2s/tgt-traink5.txt')

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
