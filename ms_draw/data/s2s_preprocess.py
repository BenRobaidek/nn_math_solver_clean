"""
s2s preprocessor for tencent
"""
def main():
    # WITH SNI

    # WITHOUT SNI
    split('./working/no_sni/train.tsv',  './working/s2s_nosni/src-train.txt',  './working/s2s_nosni/tgt-train.txt')
    split('./working/no_sni/val.tsv',    './working/s2s_nosni/src-val.txt',    './working/s2s_nosni/tgt-val.txt')
    split('./working/no_sni/test.tsv',   './working/s2s_nosni/src-test.txt',   './working/s2s_nosni/tgt-test.txt')

def split(tsv, src_txt, tgt_txt):
    tsv = open(tsv).readlines()
    with open(src_txt, 'w') as src_txt:
        for x in tsv:
            src_txt.write(x.split('\t')[0].strip() + '\n')
    with open(tgt_txt, 'w') as tgt_txt:
        for x in tsv:
            tgt_txt.write(x.split('\t')[1].strip() + '\n')

if __name__ == '__main__':
    main()
