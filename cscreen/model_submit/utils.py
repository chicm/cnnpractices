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
    
def create_res101(load_weights=False):
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.cuda()

    w_file = MODEL_DIR + '/res101.pth'
    if load_weights:
        load_weights_file(model_ft, w_file)

    return model_ft, w_file

def create_res152(load_weights=False):
    res152 = models.resnet152(pretrained=True)
    num_ftrs = res152.fc.in_features
    res152.fc = nn.Linear(num_ftrs, 3)
    res152 = res152.cuda()

    w_file = MODEL_DIR + '/res152.pth'
    if load_weights:
        load_weights_file(res152, w_file)
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

    return desnet_ft, w_file
