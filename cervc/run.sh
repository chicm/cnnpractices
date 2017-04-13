#!/bin/sh

#python cervc.py --mb
#python cervc.py --createval
python cervc.py --gentrainfeats
python cervc.py --train
python cervc.py --gentestfeats
python cervc.py --predict
