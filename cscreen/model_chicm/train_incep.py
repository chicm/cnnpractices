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
from utils import create_inceptionv3

data_dir = settings.RESIZED_DATA_PATH

RESULT_DIR = data_dir + '/results'
CLASSES_FILE = RESULT_DIR + '/train_classes.dat'
MODEL_DIR = settings.MODEL_PATH
batch_size = 16
epochs = 100

data_transforms = {
    'train': transforms.Compose([
        #transforms.Scale(320), # for vgg19
        transforms.RandomSizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Scale(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'valid']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'valid']}


dset_sizes = {x: len(dsets[x]) for x in ['train', 'valid']}
dset_classes = dsets['train'].classes
save_array(CLASSES_FILE, dset_classes)

w_files_training = []

def save_weights(acc, model, epoch, max_num=2):
    f_name = '{}_{}_{:.5f}_.pth'.format(model.name, epoch, acc)
    w_file_path = os.path.join(MODEL_DIR, f_name)
    if len(w_files_training) < max_num:
        w_files_training.append((acc, w_file_path))
        torch.save(model.state_dict(), w_file_path)
        return
    min = 10.0
    index_min = -1
    for i, item in enumerate(w_files_training):
        val_acc, fp = item
        if min > val_acc:
            index_min = i
            min = val_acc
    #print(min)
    if acc > min:
        torch.save(model.state_dict(), w_file_path)
        os.remove(w_files_training[index_min][1])
        w_files_training[index_min] = (acc, w_file_path)

def train_model(model, criterion, optimizer, w_file, lr_scheduler, max_num = 2, init_lr=0.001, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0
    loader = dset_loaders
    print(model.name)
    
    for epoch in range(num_epochs):
        epoch_since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch, init_lr=init_lr)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                #print(labels)

                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid':
                save_weights(epoch_acc, model, epoch, max_num=max_num)
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model = copy.deepcopy(model)
                #torch.save(best_model.state_dict(), w_file)

        print('epoch {}: {:.0f}s'.format(epoch, time.time()-epoch_since))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print(w_files_training)
    return best_model

def cyc_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    if epoch % lr_decay_epoch == 0 and epoch >= lr_decay_epoch:
        lr = lr * 0.6
    if lr < 5e-6:
        lr = 0.0001
    
    if epoch % lr_decay_epoch == 0 and epoch >= lr_decay_epoch:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return optimizer    

def train(model, w_file, init_lr = 0.001, num_epochs = epochs, max_num = 2):
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)

    model = train_model(model, criterion, optimizer_ft, w_file, cyc_lr_scheduler, init_lr=init_lr, 
                        num_epochs=num_epochs, max_num = max_num)
    return model

def train_inception_v3():
    print('training inception_v3')
    model, w_file = create_inceptionv3()
    try:
        model.load_state_dict(torch.load(w_file))
    except:
        print('{} not found, continue'.format(w_file))
        pass
    train(model, w_file)

parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs=1, help="train model")

args = parser.parse_args()
if args.train:
    print('start training model')
    mname = args.train[0]
    if mname == 'inceptionv3':
        train_inception_v3()
    else:
        print('model name {} not found'.format(mname))
    print('done')

