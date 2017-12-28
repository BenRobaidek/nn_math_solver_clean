#!/bin/bash

python ../OpenNMT-py/translate.py -gpu 0 -model ./models/best_folds_rm -src ./data/src-test.txt -tgt ./data/tgt-test.txt -replace_unk -verbose -output ./pred_folds_rm.txt
