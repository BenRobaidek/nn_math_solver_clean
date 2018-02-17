"""
Hyperparam search for onmt
"""
import sys
import os
import subprocess
import random
import itertools
import onmt
from onmt import train, opts
from tencent import solver

def main():


    rand = True

    mf = (1,)
    net_type = ('lstm',)# 'gru')
    #lr = (.001, .002)
    epochs = 10,
    bs = 8,
    opt = ('adam',)
    num_lay =  (1,)
    hs = (200,)
    num_dir = (2,)
    embdim = (128,)
    embfix = (False,)#True)
    ptemb = (False,)#True)
    dropout = (0.5,)
    save = False

    x = list(itertools.product(net_type, epochs, bs, opt, num_lay, hs, num_dir,
                                                embdim, embfix, ptemb, dropout, mf))
    if rand: random.shuffle(x)

    for (net_type, epoch, bs, opt, num_lay, hs, num_dir, embdim, embfix, ptemb,
                                                            dropout, mf) in x:
        print(('Training s2s: (net_type=%s, epoch=%d, bs=%d, opt=%s, ' + \
                'num_lay=%d, hs=%d, num_dir=%d, embdim=%d, embfix=%s, ' + \
                'ptemb=%s, dropout=%.1f, mf=%d})') %
            (net_type, epoch, bs, opt, num_lay, hs, num_dir, embdim, embfix,
                                                        ptemb, dropout, mf))

        #model = onmt.train.


if __name__ == '__main__':
    main()
