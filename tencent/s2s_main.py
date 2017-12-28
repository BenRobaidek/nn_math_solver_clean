import sys
import os
import subprocess
import random
import itertools
#import train
import torch

#print('Current Device:', torch.cuda.current_device())
#device = 1#int(input("Which GPU? "))
#torch.cuda.set_device(1)
#print('Current Device:', torch.cuda.current_device())

rand = True

mf = (1,)
net_type = ('lstm',)
#lr = (.001, .002)
epochs = 100,
bs = 64,
opt = ('adamax',)
num_lay =  (2,)
hs = (100, 300, 500, 1000)
num_dir = 2,
embdim = (50,)
embfix = (False,)#True)
ptemb = (False,)#True)
dropout = (.3,)



x = list(itertools.product(net_type, epochs, bs, opt, num_lay, hs, num_dir,
                                            embdim, embfix, ptemb, dropout, mf))
if rand: random.shuffle(x)
try:
    for (net_type, epoch, bs, opt, num_lay, hs, num_dir, embdim, embfix, ptemb,
                                                            dropout, mf) in x:
        if not (embfix and not ptemb):
            print('Training: -hsz=' + str(hs))
            os.system('python ../rik_mathsolver/s2s_bland.py' + \
                        ' -hsz=' + str(hs))

except(KeyboardInterrupt, SystemExit):
    sys.exit("Interrupted by ctrl+c\n")
