"""
s2s preprocessor for tencent
"""
def main():
    # WITH SNI
    split('./working/basic/train.tsv',  './working/s2s/src-train.txt',  './working/s2s/tgt-train.txt')
    split('./working/basic/val.tsv',    './working/s2s/src-val.txt',    './working/s2s/tgt-val.txt')
    split('./working/basic/test.tsv',   './working/s2s/src-test.txt',   './working/s2s/tgt-test.txt')

    split('./working/basic/traink1234.tsv', './working/s2s/src-traink1234.txt', './working/s2s/tgt-traink1234.txt')
    split('./working/basic/valk1234.tsv',   './working/s2s/src-valk1234.txt',   './working/s2s/tgt-valk1234.txt')
    split('./working/basic/testk5.tsv',     './working/s2s/src-testk5.txt',     './working/s2s/tgt-testk5.txt')

    split('./working/basic/traink2345.tsv', './working/s2s/src-traink2345.txt', './working/s2s/tgt-traink2345.txt')
    split('./working/basic/valk2345.tsv',   './working/s2s/src-valk2345.txt',   './working/s2s/tgt-valk2345.txt')
    split('./working/basic/testk1.tsv',     './working/s2s/src-testk1.txt',     './working/s2s/tgt-testk1.txt')

    split('./working/basic/traink3451.tsv', './working/s2s/src-traink3451.txt', './working/s2s/tgt-traink3451.txt')
    split('./working/basic/valk3451.tsv',   './working/s2s/src-valk3451.txt',   './working/s2s/tgt-valk3451.txt')
    split('./working/basic/testk2.tsv',     './working/s2s/src-testk2.txt',     './working/s2s/tgt-testk2.txt')

    split('./working/basic/traink4512.tsv', './working/s2s/src-traink4512.txt', './working/s2s/tgt-traink4512.txt')
    split('./working/basic/valk4512.tsv',   './working/s2s/src-valk4512.txt',   './working/s2s/tgt-valk4512.txt')
    split('./working/basic/testk3.tsv',     './working/s2s/src-testk3.txt',     './working/s2s/tgt-testk3.txt')

    split('./working/basic/traink5123.tsv', './working/s2s/src-traink5123.txt', './working/s2s/tgt-traink5123.txt')
    split('./working/basic/valk5123.tsv',   './working/s2s/src-valk5123.txt',   './working/s2s/tgt-valk5123.txt')
    split('./working/basic/testk4.tsv',     './working/s2s/src-testk4.txt',     './working/s2s/tgt-testk4.txt')

    # WITHOUT SNI
    split('./working/no_sni/train.tsv',  './working/s2s/src-train.txt',  './working/s2s/tgt-train.txt')
    split('./working/no_sni/val.tsv',    './working/s2s/src-val.txt',    './working/s2s/tgt-val.txt')
    split('./working/no_sni/test.tsv',   './working/s2s/src-test.txt',   './working/s2s/tgt-test.txt')

    split('./working/no_sni/traink1234.tsv', './working/s2s/src-traink1234.txt', './working/s2s/tgt-traink1234.txt')
    split('./working/no_sni/valk1234.tsv',   './working/s2s/src-valk1234.txt',   './working/s2s/tgt-valk1234.txt')
    split('./working/no_sni/testk5.tsv',     './working/s2s/src-testk5.txt',     './working/s2s/tgt-testk5.txt')

    split('./working/no_sni/traink2345.tsv', './working/s2s/src-traink2345.txt', './working/s2s/tgt-traink2345.txt')
    split('./working/no_sni/valk2345.tsv',   './working/s2s/src-valk2345.txt',   './working/s2s/tgt-valk2345.txt')
    split('./working/no_sni/testk1.tsv',     './working/s2s/src-testk1.txt',     './working/s2s/tgt-testk1.txt')

    split('./working/no_sni/traink3451.tsv', './working/s2s/src-traink3451.txt', './working/s2s/tgt-traink3451.txt')
    split('./working/no_sni/valk3451.tsv',   './working/s2s/src-valk3451.txt',   './working/s2s/tgt-valk3451.txt')
    split('./working/no_sni/testk2.tsv',     './working/s2s/src-testk2.txt',     './working/s2s/tgt-testk2.txt')

    split('./working/no_sni/traink4512.tsv', './working/s2s/src-traink4512.txt', './working/s2s/tgt-traink4512.txt')
    split('./working/no_sni/valk4512.tsv',   './working/s2s/src-valk4512.txt',   './working/s2s/tgt-valk4512.txt')
    split('./working/no_sni/testk3.tsv',     './working/s2s/src-testk3.txt',     './working/s2s/tgt-testk3.txt')

    split('./working/no_sni/traink5123.tsv', './working/s2s/src-traink5123.txt', './working/s2s/tgt-traink5123.txt')
    split('./working/no_sni/valk5123.tsv',   './working/s2s/src-valk5123.txt',   './working/s2s/tgt-valk5123.txt')
    split('./working/no_sni/testk4.tsv',     './working/s2s/src-testk4.txt',     './working/s2s/tgt-testk4.txt')

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
