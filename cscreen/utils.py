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

MODEL_DIR = settings.MODEL_PATH

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def load_weights_file(model, w_file):
    model.load_state_dict(torch.load(w_file))

def create_res50(load_weights=False):
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.cuda()

    w_file = MODEL_DIR + '/res50.pth'
    if load_weights:
        load_weights_file(model_ft, w_file)
    
    model_ft.name = 'res50'
    return model_ft, w_file

def create_res101(load_weights=False):
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.cuda()

    w_file = MODEL_DIR + '/res101.pth'
    if load_weights:
        load_weights_file(model_ft, w_file)

    model_ft.name = 'res101'
    return model_ft, w_file

def create_res152(load_weights=False):
    res152 = models.resnet152(pretrained=True)
    num_ftrs = res152.fc.in_features
    res152.fc = nn.Linear(num_ftrs, 3)
    res152 = res152.cuda()

    w_file = MODEL_DIR + '/res152.pth'
    if load_weights:
        load_weights_file(res152, w_file)
    res152.name = 'res152'
    return res152, w_file

def create_dense161(load_weights=False):
    desnet_ft = models.densenet161(pretrained=True)
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Linear(num_ftrs, 3)
    print(num_ftrs)
    desnet_ft = desnet_ft.cuda()
    w_file = MODEL_DIR + '/dense161.pth'

    if load_weights:
        load_weights_file(desnet_ft, w_file)
    desnet_ft.name = 'dense161'
    return desnet_ft, w_file

def create_dense169(load_weights=False):
    desnet_ft = models.densenet169(pretrained=True)
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Linear(num_ftrs, 3)
    print(num_ftrs)
    desnet_ft = desnet_ft.cuda()
    w_file = MODEL_DIR + '/dense169.pth'

    if load_weights:
        load_weights_file(desnet_ft, w_file)
    desnet_ft.name = 'dense169'
    return desnet_ft, w_file

def create_dense201(load_weights=False):
    desnet_ft = models.densenet201(pretrained=True)
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Linear(num_ftrs, 3)
    print(num_ftrs)
    desnet_ft = desnet_ft.cuda()
    w_file = MODEL_DIR + '/dense201.pth'

    if load_weights:
        load_weights_file(desnet_ft, w_file)
    desnet_ft.name = 'dense201'
    return desnet_ft, w_file

def create_vgg19bn(load_weights=False):
    vgg19_bn_ft = models.vgg19_bn(num_classes=3).cuda()
    w_file = MODEL_DIR + '/vgg19bn.pth'

    if load_weights:
        load_weights_file(vgg19_bn_ft, w_file)
    vgg19_bn_ft.name = 'vgg19bn'
    return vgg19_bn_ft, w_file

def create_inceptionv3(load_weights=False):
    incept_ft = models.inception_v3(pretrained=True)
    num_ftrs = incept_ft.fc.in_features
    incept_ft.fc = nn.Linear(num_ftrs, 3)
    incept_ft.aux_logits=False
    print(num_ftrs)
    incept_ft = incept_ft.cuda()
    w_file = MODEL_DIR + '/inceptionv3.pth'

    if load_weights:
        load_weights_file(incept_ft, w_file)
    incept_ft.name = 'inceptionv3'

    return incept_ft, w_file

def create_vgg19(load_weights=False):
    vgg19_ft = models.vgg19(pretrained=True)
    vgg19_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3))
    vgg19_ft = vgg19_ft.cuda()

    w_file = MODEL_DIR + '/vgg19.pth'

    if load_weights:
        load_weights_file(vgg19_ft, w_file)
    vgg19_ft.name = 'vgg19'
    return vgg19_ft, w_file

def create_vgg16(load_weights=False):
    vgg16_ft = models.vgg16(pretrained=True)
    vgg16_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3))
    vgg16_ft = vgg16_ft.cuda()

    w_file = MODEL_DIR + '/vgg16.pth'

    if load_weights:
        load_weights_file(vgg16_ft, w_file)
    vgg16_ft.name = 'vgg16'
    return vgg16_ft, w_file