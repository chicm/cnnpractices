#!/bin/sh

python predict.py --ens
python predict.py --sub submit1.csv 0.98
