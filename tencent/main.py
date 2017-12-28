import sys
import os
import subprocess
import random
import itertools
import torch

#print('Current Device:', torch.cuda.current_device())
#device = 1#int(input("Which GPU? "))
#torch.cuda.set_device(1)
#print('Current Device:', torch.cuda.current_device())

rand = True

common_data = (0.2,0.4,0.6,0.8)

mf = (1,2)
net_type = ('lstm',)
#lr = (.001, .002)
epochs = 100,
bs = 8, 16, 64
opt = ('adamax',)# 'adam', 'sgd')
num_lay =  (1, 2)
hs = (100, 300, 500)
num_dir = 2,
embdim = (50, 100, 200, 300, 500)
embfix = (False,)#True)
ptemb = (False,)#True)
dropout = (0, .3, .5, .7)



x = list(itertools.product(common_data,net_type, epochs, bs, opt, num_lay, hs, num_dir,
                                            embdim, embfix, ptemb, dropout, mf))
if rand: random.shuffle(x)
try:
    for (common_data,net_type, epoch, bs, opt, num_lay, hs, num_dir, embdim, embfix, ptemb,
                                                            dropout, mf) in x:
        if not (embfix and not ptemb):
            print(('Training: (net_type=%s, epoch=%d, bs=%d, opt=%s, ' + \
                    'num_lay=%d, hs=%d, num_dir=%d, embdim=%d, embfix=%s, ' + \
                    'ptemb=%s, dropout=%.1f, mf=%d})') %
                (net_type, epoch, bs, opt, num_lay, hs, num_dir, embdim, embfix,
                                                            ptemb, dropout, mf))
            os.system('python ../model/train.py' + \
                        ' -save-path=' + './models/'+ \
                        ' -data-path=' + './data/'+ \
                        ' -train-path=' + 'train_' + str(common_data) + '.tsv' + \
                        ' -dev-path=' + 'dev_' + str(common_data) + '.tsv' + \
                        ' -test-path=' + 'test.tsv' + \
                        ' -net-type=' + str(net_type) + \
                        #' -lr=' + str(lr) + \
                        ' -epochs=' + str(epochs[0]) + \
                        ' -batch-size=' + str(bs) + \
                        ' -opt=' + opt + \
                        ' -num-layers=' + str(num_lay) + \
                        ' -hidden-sz=' + str(hs) + \
                        ' -num-dir=' + str(num_dir) + \
                        ' -emb-dim=' + str(embdim) + \
                        ' -embfix=' + str(embfix) + \
                        ' -pretr-emb=' + str(ptemb) + \
                        ' -dropout=' + str(dropout) + \
                        ' -mf=' + str(mf) + \
                        ' -folder=' + 'common' + str(common_data) + '/')
            os.system('sort -o ./models/common' + str(common_data) + '/best_models.txt'
                                './models/common' + str(common_data) + '/best_models.txt')
except(KeyboardInterrupt, SystemExit):
    sys.exit("Interrupted by ctrl+c\n")
