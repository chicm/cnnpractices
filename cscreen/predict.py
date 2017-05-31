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
import os
import cv2
import random
import argparse
import bcolz
import pandas as pd
import random
from PIL import Image
from utils import save_array, load_array
from utils import create_dense161, create_dense201, create_res101, create_res152

data_dir = settings.DATA_DIR
test_dir = settings.TEST_DATA_PATH

RESULT_DIR = data_dir + '/results'
PRED_FILE = RESULT_DIR + '/pred_ens_torch.dat'
PRED_FILE_RAW = RESULT_DIR + '/pred_ens_raw.dat'
CLASSES_FILE = RESULT_DIR + '/train_classes.dat'
batch_size = 16

data_transforms = {
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

dsets = datasets.ImageFolder(test_dir, data_transforms['test'])
dsets.imgs = sorted(dsets.imgs)
         
dset_classes = load_array(CLASSES_FILE)
print(dset_classes)

test_loader = torch.utils.data.DataLoader(dsets, batch_size=batch_size,
                                               shuffle=False, num_workers=4)

use_gpu = torch.cuda.is_available()

def make_preds(net, test_loader):
    preds = []
    m = nn.Softmax()
    for i, (img, indices) in enumerate(test_loader, 0):
        inputs = Variable(img.cuda())
        outputs = net(inputs)
        pred = m(outputs).data.cpu().tolist()
        for p in pred:
            preds.append(p)
    return preds

def ensemble():
    res101, _ = create_res101(True)
    res152, _ = create_res152(True)
    dense201, _ = create_dense201(True)
    dense161, _ = create_dense161(True)
    
    pred1 = np.array(make_preds(res101, test_loader))
    pred2 = np.array(make_preds(res152, test_loader))
    pred3 = np.array(make_preds(dense201, test_loader))
    pred4 = np.array(make_preds(dense161, test_loader))
    raw_preds = [pred1, pred2, pred3, pred4]
    save_array(PRED_FILE_RAW, raw_preds)
    preds = np.mean(raw_preds, axis=0)
    save_array(PRED_FILE, preds)
    print(preds[:10])

def do_clip(arr, mx): 
    return np.clip(arr, (1-mx)/2, mx)

def submit(filename, clip):
    filenames = [f.split('/')[-1] for f, i in dsets.imgs]

    preds = load_array(PRED_FILE)
    subm = do_clip(preds, clip)
    subm_name = RESULT_DIR+'/'+filename  
    submission = pd.DataFrame(subm, columns=dset_classes)
    submission.insert(0, 'image_name', filenames)
    print(submission.head())
    submission.to_csv(subm_name, index=False)

parser = argparse.ArgumentParser()
parser.add_argument("--ens", action='store_true', help="ensemble predict")
parser.add_argument("--sub", nargs=2, help="generate submission file")

args = parser.parse_args()
if args.ens:
    ensemble()
    print('done')
if args.sub:
    print('generating submision file...')
    submit(args.sub[0], (float)(args.sub[1]))
    print('done')
