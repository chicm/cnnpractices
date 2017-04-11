#!/bin/sh

python statefarm2.py --mb
python statefarm2.py --createval
python statefarm2.py --gentrainfeats
python statefarm2.py --train
python statefarm2.py --gentestfeats
python statefarm2.py --predict
