#!/bin/sh

python train.py --train vgg19bn
python train.py --train vgg16bn
python train.py --train dense161
python train.py --train dense201
python train.py --train dense169
python train.py --train res50
python train.py --train res101
python train.py --train res152
python train_incep.py --train inceptionv3
