#!/bin/sh

python train.py --train dense161
python train.py --train dense201
python train.py --train res101
python train.py --train res152
