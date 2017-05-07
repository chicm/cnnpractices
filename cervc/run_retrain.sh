#!/bin/sh

python retrain.py --gentrainfeats
python retrain.py --train
python retrain.py --gentestfeats
