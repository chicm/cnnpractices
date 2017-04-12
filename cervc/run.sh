#!/bin/sh

#python statefarm2.py --mb
#python statefarm2.py --createval
python cervc.py --gentrainfeats
python cervc.py --train
python cervc.py --gentestfeats
python cervc.py --predict
