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
from utils import save_array, load_array
from utils import create_dense161, create_dense201, create_res101, create_res152
from utils import create_dense169, create_res50, create_vgg16, create_vgg19, create_inceptionv3
from functools import cmp_to_key

data_dir = settings.DATA_DIR
test_dir = settings.TEST_DATA_PATH
MODEL_DIR = settings.MODEL_PATH

RESULT_DIR = data_dir + '/results'
PRED_FILE = RESULT_DIR + '/pred_ens_torch.dat'
PRED_FILE_RAW = RESULT_DIR + '/pred_ens_raw.dat'
CLASSES_FILE = RESULT_DIR + '/train_classes.dat'
batch_size = 16

w_file_matcher = ['dense161*pth', 'dense201*pth','dense169*pth','inceptionv3*pth',
    'res50*pth','res101*pth', 'res152*pth', 'vgg16*pth', 'vgg19*pth']

data_transforms = {
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'testv3': transforms.Compose([
        transforms.Scale(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def mycmp(one, two):
    num1 = int(one[0].split('/')[-1].split('.')[0])
    num2 = int(two[0].split('/')[-1].split('.')[0])
    return num1 - num2

dsets = datasets.ImageFolder(test_dir, data_transforms['test'])
#dsets.imgs = sorted(dsets.imgs, key=cmp_to_key(mycmp))
dsets.imgs = sorted(dsets.imgs)

dsetsv3 = datasets.ImageFolder(test_dir, data_transforms['testv3'])
#dsetsv3.imgs = sorted(dsetsv3.imgs, key=cmp_to_key(mycmp))
dsetsv3.imgs = sorted(dsetsv3.imgs)

print(dsets.imgs[:5])
print(dsetsv3.imgs[:5])
         
dset_classes = load_array(CLASSES_FILE)
print(dset_classes)

test_loader = torch.utils.data.DataLoader(dsets, batch_size=batch_size,
                                               shuffle=False, num_workers=4)

test_v3_loader = torch.utils.data.DataLoader(dsetsv3, batch_size=batch_size,
                                               shuffle=False, num_workers=4)

use_gpu = torch.cuda.is_available()

def make_preds(net, test_loader):
    loader = test_loader
    if hasattr(net, 'name') and net.name == 'inceptionv3':
        print('making prediction with inceptioinv3')
        loader = test_v3_loader
    preds = []
    m = nn.Softmax()
    for i, (img, indices) in enumerate(loader, 0):
        inputs = Variable(img.cuda())
        outputs = net(inputs)
        pred = m(outputs).data.cpu().tolist()
        for p in pred:
            preds.append(p)
    return preds

def ensemble():
    preds_raw = []
    os.chdir(MODEL_DIR)
    for match_str in w_file_matcher:
        w_files = glob.glob(match_str)
        for w_file in w_files:
            full_w_file = MODEL_DIR + '/' + w_file
            if w_file.startswith('dense161'):
                model, _ = create_dense161()
            elif w_file.startswith('dense169'):
                model, _ = create_dense169()
            elif w_file.startswith('dense201'):
                model, _ = create_dense201()
            elif w_file.startswith('res50'):
                model,_ = create_res50()
            elif w_file.startswith('res101'):
                model,_ = create_res101()
            elif w_file.startswith('res152'):
                model,_ = create_res152()
            elif w_file.startswith('vgg16'):
                model,_ = create_vgg16()
            elif w_file.startswith('vgg19'):
                model,_ = create_vgg19()
            elif w_file.startswith('inceptionv3'):
                model,_ = create_inceptionv3()
            else:
                pass
            model.load_state_dict(torch.load(full_w_file))
            print(full_w_file)

            pred = make_preds(model, test_loader)
            pred = np.array(pred)
            preds_raw.append(pred)

            del model
            
    save_array(PRED_FILE_RAW, preds_raw)
    preds = np.mean(preds_raw, axis=0)
    save_array(PRED_FILE, preds)

def do_clip(arr, mx): 
    return np.clip(arr, (1-mx)/2, mx)

def submit(filename, clip):
    filenames = [f.split('/')[-1] for f, i in dsets.imgs]

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
