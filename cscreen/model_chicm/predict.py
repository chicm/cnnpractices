import settings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os, glob
import cv2
import random
import argparse
import bcolz
import pandas as pd
import random
from PIL import Image
from utils import save_array, load_array, get_acc_from_w_filename
from utils import create_dense161, create_dense201, create_res101, create_res152, create_dense121
from utils import create_dense169, create_res50, create_inceptionv3
from utils import create_vgg16, create_vgg19, create_vgg16bn, create_vgg19bn
from cscreendataset import get_stage1_test_loader, get_stage2_test_loader

data_dir = settings.RESIZED_DATA_PATH
MODEL_DIR = settings.MODEL_PATH

RESULT_DIR = data_dir + '/results'
PRED_FILE = RESULT_DIR + '/pred_ens_torch.dat'
PRED_FILE_WEIGHTED = RESULT_DIR + '/pred_ens_weighted.dat'
PRED_FILE_RAW = RESULT_DIR + '/pred_ens_raw.dat'
PRED_FILE_RAW_WEIGHTED = RESULT_DIR + '/pred_raw_weighted.dat'
CLASSES_FILE = RESULT_DIR + '/train_classes.dat'
TEST_FILE_NAMES = RESULT_DIR + '/filenames.dat'
batch_size = 16

w_file_matcher = ['dense161*pth', 'dense201*pth','dense169*pth','dense121*pth','inceptionv3*pth',
    'res50*pth','res101*pth', 'res152*pth', 'vgg16*pth', 'vgg19*pth']

         
dset_classes = load_array(CLASSES_FILE)
print(dset_classes)

def make_preds(net, stage):
    if stage == 'stage1':
        loader = get_stage1_test_loader(net.name)
    elif stage == 'stage2':
        loader = get_stage2_test_loader(net.name)
    else:
        print('Wrong stage name:{}'.format(stage))
        exit()
    preds = []
    m = nn.Softmax()
    for i, (img, _) in enumerate(loader, 0):
        inputs = Variable(img.cuda())
        outputs = net(inputs)
        pred = m(outputs).data.cpu().tolist()
        for p in pred:
            preds.append(p)
    return preds, loader.filenames

def get_weight(acc):
    if acc < 0.78:
        return 0
    elif acc < 0.8:
        return 0.2
    elif acc < 0.81:
        return 0.4
    elif acc < 0.82:
        return 0.5
    elif acc < 0.83:
        return 0.6
    elif acc < 0.84:
        return 0.7
    elif acc < 0.85:
        return 0.8
    elif acc < 0.86:
        return 0.9
    elif acc < 0.87:
        return 1.0
    elif acc < 0.88:
        return 1.1
    elif acc < 0.89:
        return 1.2
    elif acc >= 0.89:
        return 1.3

def ensemble(stage):
    preds_weighted = None
    total_weight = 0
    preds_raw = []
    preds_raw_weighted = []
    
    for match_str in w_file_matcher:
        #print(match_str)
        os.chdir(MODEL_DIR)
        w_files = glob.glob(match_str)
        #print('cur:' + os.getcwd())
        for w_file in w_files:
            full_w_file = MODEL_DIR + '/' + w_file
            print(full_w_file)
            if w_file.startswith('dense161'):
                model = create_dense161()
            elif w_file.startswith('dense169'):
                model = create_dense169()
            elif w_file.startswith('dense201'):
                model = create_dense201()
            elif w_file.startswith('dense121'):
                model = create_dense121()
            elif w_file.startswith('res50'):
                model = create_res50()
            elif w_file.startswith('res101'):
                model = create_res101()
            elif w_file.startswith('res152'):
                model = create_res152()
            elif w_file.startswith('vgg16bn'):
                model = create_vgg16bn()
            elif w_file.startswith('vgg19bn'):
                model = create_vgg19bn()
            elif w_file.startswith('vgg19'):
                model = create_vgg19()
            elif w_file.startswith('inceptionv3'):
                model = create_inceptionv3()
            else:
                print('No model for {}'.format(full_w_file))
                continue
            model.load_state_dict(torch.load(full_w_file))

            pred, filenames = make_preds(model, stage)
            pred = np.array(pred)
            preds_raw.append(pred)
            weight = get_weight(get_acc_from_w_filename(full_w_file))
            preds_raw_weighted.append((pred, weight))

            if preds_weighted is None:
                preds_weighted = np.zeros(pred.shape)
            preds_weighted = preds_weighted + pred * weight
            total_weight += weight

            del model
            
    preds_weighted = preds_weighted/total_weight

    save_array(PRED_FILE_RAW, preds_raw)
    save_array(PRED_FILE_RAW_WEIGHTED, preds_raw_weighted)
    preds = np.mean(preds_raw, axis=0)
    save_array(PRED_FILE, preds)
    save_array(PRED_FILE_WEIGHTED, preds_weighted)
    save_array(TEST_FILE_NAMES, filenames)

def do_clip(arr, mx): 
    return np.clip(arr, (1-mx)/2, mx)

def submit(filename, clip, use_weight=False):
    #filenames = [f.split('/')[-1] for f, i in dsets.imgs]
    #filenames = get_stage1_test_loader('res50').filenames
    filenames = load_array(TEST_FILE_NAMES)
    if use_weight:
        preds = load_array(PRED_FILE_WEIGHTED)
    else:
        preds = load_array(PRED_FILE)
    if clip > 0.9999:
        subm = np.array(preds)
    else:
        subm = do_clip(preds, clip)
    subm_name = RESULT_DIR+'/'+filename  
    submission = pd.DataFrame(subm, columns=dset_classes)
    submission.insert(0, 'image_name', filenames)
    print(submission.head())
    submission.to_csv(subm_name, index=False)

parser = argparse.ArgumentParser()
parser.add_argument("--ens", nargs=1, help="ensemble predict")
parser.add_argument("--sub", nargs=2, help="generate submission file")

args = parser.parse_args()
if args.ens:
    ensemble(args.ens[0])
    print('done')
if args.sub:
    print('generating submision file...')
    submit(args.sub[0], (float)(args.sub[1]), True)
    print('done')
    print('Please find submisson file at: {}'.format(RESULT_DIR+'/'+args.sub[0]))
