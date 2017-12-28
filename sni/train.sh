#!/bin/bash

python ../OpenNMT-py/train.py -data ./data/Math23K -save_model ./models/Math23K-model -gpuid 0 -epochs=50
