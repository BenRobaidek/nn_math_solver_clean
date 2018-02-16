"""
s2s preprocessor for tencent
"""
def main():
    # WITH SNI
    split('./working/basic/train.tsv',  './working/s2s_basic/src-train.txt',  './working/s2s_basic/tgt-train.txt')
    split('./working/basic/val.tsv',    './working/s2s_basic/src-val.txt',    './working/s2s_basic/tgt-val.txt')
    split('./working/basic/test.tsv',   './working/s2s_basic/src-test.txt',   './working/s2s_basic/tgt-test.txt')

    split('./working/basic/traink1234.tsv', './working/s2s_basic/src-traink1234.txt', './working/s2s_basic/tgt-traink1234.txt')
    split('./working/basic/valk1234.tsv',   './working/s2s_basic/src-valk1234.txt',   './working/s2s_basic/tgt-valk1234.txt')
    split('./working/basic/testk5.tsv',     './working/s2s_basic/src-testk5.txt',     './working/s2s_basic/tgt-testk5.txt')

    split('./working/basic/traink2345.tsv', './working/s2s_basic/src-traink2345.txt', './working/s2s_basic/tgt-traink2345.txt')
    split('./working/basic/valk2345.tsv',   './working/s2s_basic/src-valk2345.txt',   './working/s2s_basic/tgt-valk2345.txt')
    split('./working/basic/testk1.tsv',     './working/s2s_basic/src-testk1.txt',     './working/s2s_basic/tgt-testk1.txt')

    split('./working/basic/traink3451.tsv', './working/s2s_basic/src-traink3451.txt', './working/s2s_basic/tgt-traink3451.txt')
    split('./working/basic/valk3451.tsv',   './working/s2s_basic/src-valk3451.txt',   './working/s2s_basic/tgt-valk3451.txt')
    split('./working/basic/testk2.tsv',     './working/s2s_basic/src-testk2.txt',     './working/s2s_basic/tgt-testk2.txt')

    split('./working/basic/traink4512.tsv', './working/s2s_basic/src-traink4512.txt', './working/s2s_basic/tgt-traink4512.txt')
    split('./working/basic/valk4512.tsv',   './working/s2s_basic/src-valk4512.txt',   './working/s2s_basic/tgt-valk4512.txt')
    split('./working/basic/testk3.tsv',     './working/s2s_basic/src-testk3.txt',     './working/s2s_basic/tgt-testk3.txt')

    split('./working/basic/traink5123.tsv', './working/s2s_basic/src-traink5123.txt', './working/s2s_basic/tgt-traink5123.txt')
    split('./working/basic/valk5123.tsv',   './working/s2s_basic/src-valk5123.txt',   './working/s2s_basic/tgt-valk5123.txt')
    split('./working/basic/testk4.tsv',     './working/s2s_basic/src-testk4.txt',     './working/s2s_basic/tgt-testk4.txt')

    # WITHOUT SNI
    split('./working/no_sni/train.tsv',  './working/s2s_wosni/src-train.txt',  './working/s2s_wosni/tgt-train.txt')
    split('./working/no_sni/val.tsv',    './working/s2s_wosni/src-val.txt',    './working/s2s_wosni/tgt-val.txt')
    split('./working/no_sni/test.tsv',   './working/s2s_wosni/src-test.txt',   './working/s2s_wosni/tgt-test.txt')

    split('./working/no_sni/traink1234.tsv', './working/s2s_wosni/src-traink1234.txt', './working/s2s_wosni/tgt-traink1234.txt')
    split('./working/no_sni/valk1234.tsv',   './working/s2s_wosni/src-valk1234.txt',   './working/s2s_wosni/tgt-valk1234.txt')
    split('./working/no_sni/testk5.tsv',     './working/s2s_wosni/src-testk5.txt',     './working/s2s_wosni/tgt-testk5.txt')

    split('./working/no_sni/traink2345.tsv', './working/s2s_wosni/src-traink2345.txt', './working/s2s_wosni/tgt-traink2345.txt')
    split('./working/no_sni/valk2345.tsv',   './working/s2s_wosni/src-valk2345.txt',   './working/s2s_wosni/tgt-valk2345.txt')
    split('./working/no_sni/testk1.tsv',     './working/s2s_wosni/src-testk1.txt',     './working/s2s_wosni/tgt-testk1.txt')

    split('./working/no_sni/traink3451.tsv', './working/s2s_wosni/src-traink3451.txt', './working/s2s_wosni/tgt-traink3451.txt')
    split('./working/no_sni/valk3451.tsv',   './working/s2s_wosni/src-valk3451.txt',   './working/s2s_wosni/tgt-valk3451.txt')
    split('./working/no_sni/testk2.tsv',     './working/s2s_wosni/src-testk2.txt',     './working/s2s_wosni/tgt-testk2.txt')

    split('./working/no_sni/traink4512.tsv', './working/s2s_wosni/src-traink4512.txt', './working/s2s_wosni/tgt-traink4512.txt')
    split('./working/no_sni/valk4512.tsv',   './working/s2s_wosni/src-valk4512.txt',   './working/s2s_wosni/tgt-valk4512.txt')
    split('./working/no_sni/testk3.tsv',     './working/s2s_wosni/src-testk3.txt',     './working/s2s_wosni/tgt-testk3.txt')

    split('./working/no_sni/traink5123.tsv', './working/s2s_wosni/src-traink5123.txt', './working/s2s_wosni/tgt-traink5123.txt')
    split('./working/no_sni/valk5123.tsv',   './working/s2s_wosni/src-valk5123.txt',   './working/s2s_wosni/tgt-valk5123.txt')
    split('./working/no_sni/testk4.tsv',     './working/s2s_wosni/src-testk4.txt',     './working/s2s_wosni/tgt-testk4.txt')

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
