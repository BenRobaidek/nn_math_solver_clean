#!/bin/bash
python ../OpenNMT-py/preprocess.py -train_src ./data/src-train.txt -train_tgt ./data/tgt-train.txt -valid_src ./data/src-val.txt -valid_tgt ./data/tgt-val.txt -save_data ./data/Math23K
