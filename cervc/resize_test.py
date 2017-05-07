import os, shutil, glob, sys
import pandas as pd
import numpy as np
import cv2

DATA_DIR = '/home/chicm/data/cervc/clean640/test'
TRAIN_DIR = DATA_DIR+'/train'
TEST_DIR = DATA_DIR + '/test'
VALID_DIR = DATA_DIR + '/valid'

TGT_DIR = '/home/chicm/data/cervc/clean640/test2'
TGT_TRAIN = TGT_DIR+'/train'
TGT_TEST = TGT_DIR + '/test'
TGT_VAL = TGT_DIR + '/valid'


def resize():
    os.chdir(DATA_DIR)
    files = glob.glob('*/*.jpg')

    #drivers = sorted(driver2imgs.keys())
    #files = np.random.permutation(files)
    print(files[:10])

    for f in files:
        fn = DATA_DIR+'/'+f
        tgt_fn = TGT_DIR+'/'+f
        print(fn)
        print(tgt_fn)
        img = cv2.imread(fn)
        res = cv2.resize(img, (640, 640))
        #region = res[50:450, 50:450]
        #gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(tgt_fn, res)
        
resize()
