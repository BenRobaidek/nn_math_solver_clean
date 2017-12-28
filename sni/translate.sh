#!/bin/bash

python ../OpenNMT-py/translate.py -gpu 0 -model ./models/Math23K-model_acc_88.70_ppl_1.34_e28.pt -src ./data/src-val.txt -tgt ./data/tgt-val.txt -replace_unk -verbose -output ./pred.txt
