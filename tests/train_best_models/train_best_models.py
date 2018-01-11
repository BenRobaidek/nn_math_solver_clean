import os

def main():
    print('Training classifier on full tencent data')
    os.system('python ../../classifier/train.py' + \
                ' -save-path=' + './saved_models/classifier_basic/'+ \
                ' -save=' + 'True'
                ' -data-path=' + '../../tencent/data/working/basic/'+ \
                ' -train-path=' + 'train.tsv' + \
                ' -dev-path=' + 'val.tsv' + \
                ' -test-path=' + '../basic/test.tsv' + \
                ' -net-type=' + 'lstm' + \
                ' -epochs=' + '100' + \
                ' -batch-size=' + '64' + \
                ' -opt=' + 'adamax' + \
                ' -num-layers=' + '1' + \
                ' -hidden-sz=' + '300' + \
                ' -num-dir=' + '2' + \
                ' -emb-dim=' + '100' + \
                ' -embfix=' + 'False' + \
                ' -pretr-emb=' + 'False' + \
                ' -dropout=' + '0.3' + \
                ' -mf=' + '1' + \
                ' -folder=' + '')
    """
    print('Training classifier on common0.2 tencent data')
    os.system('python train.py' + \
    #PARAMETERS:net-lstm_e100_bs8_opt-adamax_ly1_hs300_dr2_ed300_fembFalse_ptembFalse_drp0.7_mf2
                ' -save-path=' + './saved_models/classifier_common0.2/'+ \
                ' -save=' + 'True'
                ' -data-path=' + '../../tencent/data/working/common0.2/'+ \
                ' -train-path=' + 'train.tsv' + \
                ' -dev-path=' + 'val.tsv' + \
                ' -test-path=' + '../basic/test.tsv' + \
                ' -net-type=' + '' + \
                ' -epochs=' + '100' + \
                ' -batch-size=' + '' + \
                ' -opt=' + '' + \
                ' -num-layers=' + '' + \
                ' -hidden-sz=' + '' + \
                ' -num-dir=' + '' + \
                ' -emb-dim=' + '' + \
                ' -embfix=' + '' + \
                ' -pretr-emb=' + '' + \
                ' -dropout=' + '' + \
                ' -mf=' + '' + \
                ' -folder=' + '')

    print('Training classifier on common0.4 tencent data')
    os.system('python train.py' + \
    #PARAMETERS:net-lstm_e100_bs64_opt-adamax_ly1_hs300_dr2_ed300_fembFalse_ptembFalse_drp0.0_mf2
                ' -save-path=' + './saved_models/classifier_full/'+ \
                ' -save=' + 'True'
                ' -data-path=' + '../../tencent/data/working/basic/'+ \
                ' -train-path=' + 'train.tsv' + \
                ' -dev-path=' + 'val.tsv' + \
                ' -test-path=' + '../basic/test.tsv' + \
                ' -net-type=' + '' + \
                ' -epochs=' + '100' + \
                ' -batch-size=' + '' + \
                ' -opt=' + '' + \
                ' -num-layers=' + '' + \
                ' -hidden-sz=' + '' + \
                ' -num-dir=' + '' + \
                ' -emb-dim=' + '' + \
                ' -embfix=' + '' + \
                ' -pretr-emb=' + '' + \
                ' -dropout=' + '' + \
                ' -mf=' + '' + \
                ' -folder=' + '')

    print('Training classifier on common0.6 tencent data')
    #PARAMETERS:net-lstm_e100_bs8_opt-adamax_ly1_hs100_dr2_ed100_fembFalse_ptembFalse_drp0.7_mf1
    os.system('python train.py' + \
                ' -save-path=' + './saved_models/classifier_full/'+ \
                ' -save=' + 'True'
                ' -data-path=' + '../../tencent/data/working/basic/'+ \
                ' -train-path=' + 'train.tsv' + \
                ' -dev-path=' + 'val.tsv' + \
                ' -test-path=' + '../basic/test.tsv' + \
                ' -net-type=' + '' + \
                ' -epochs=' + '100' + \
                ' -batch-size=' + '' + \
                ' -opt=' + '' + \
                ' -num-layers=' + '' + \
                ' -hidden-sz=' + '' + \
                ' -num-dir=' + '' + \
                ' -emb-dim=' + '' + \
                ' -embfix=' + '' + \
                ' -pretr-emb=' + '' + \
                ' -dropout=' + '' + \
                ' -mf=' + '' + \
                ' -folder=' + '')

    print('Training classifier on common0.8 tencent data')
    #PARAMETERS:net-lstm_e100_bs64_opt-adamax_ly1_hs300_dr2_ed500_fembFalse_ptembFalse_drp0.3_mf1
    os.system('python train.py' + \
                ' -save-path=' + './saved_models/classifier_full/'+ \
                ' -save=' + 'True'
                ' -data-path=' + '../../tencent/data/working/basic/'+ \
                ' -train-path=' + 'train.tsv' + \
                ' -dev-path=' + 'val.tsv' + \
                ' -test-path=' + '../basic/test.tsv' + \
                ' -net-type=' + '' + \
                ' -epochs=' + '100' + \
                ' -batch-size=' + '' + \
                ' -opt=' + '' + \
                ' -num-layers=' + '' + \
                ' -hidden-sz=' + '' + \
                ' -num-dir=' + '' + \
                ' -emb-dim=' + '' + \
                ' -embfix=' + '' + \
                ' -pretr-emb=' + '' + \
                ' -dropout=' + '' + \
                ' -mf=' + '' + \
                ' -folder=' + '')

    print('Training OpenNMT seq2seq on full tencent data')
    os.system('python train.py' + \
                ' -save-path=' + './saved_models/classifier_full/'+ \
                ' -save=' + 'True'
                ' -data-path=' + '../../tencent/data/working/basic/'+ \
                ' -train-path=' + 'train.tsv' + \
                ' -dev-path=' + 'val.tsv' + \
                ' -test-path=' + '../basic/test.tsv' + \
                ' -net-type=' + '' + \
                ' -epochs=' + '100' + \
                ' -batch-size=' + '' + \
                ' -opt=' + '' + \
                ' -num-layers=' + '' + \
                ' -hidden-sz=' + '' + \
                ' -num-dir=' + '' + \
                ' -emb-dim=' + '' + \
                ' -embfix=' + '' + \
                ' -pretr-emb=' + '' + \
                ' -dropout=' + '' + \
                ' -mf=' + '' + \
                ' -folder=' + '')
    """

if __name__ == '__main__':
    main()
