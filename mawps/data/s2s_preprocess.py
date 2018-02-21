"""
s2s preprocessor for tencent
"""
def main():
    # WITH SNI
    split('./train.tsv',  './working/s2s/src-train.txt',  './working/s2s/tgt-train.txt')
    split('./val.tsv',  './working/s2s/src-val.txt',  './working/s2s/tgt-val.txt')
    split('./test.tsv',  './working/s2s/src-test.txt',  './working/s2s/tgt-test.txt')

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
