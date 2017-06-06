import settings
import os, cv2, glob
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
import random
#import transforms

CUR_DIR = os.getcwd()
DATA_DIR = settings.RESIZED_DATA_PATH
TRAIN_DIR = DATA_DIR + '/train'
VAL_DIR = DATA_DIR + '/valid'

def pil_load(img_path):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class CScreenDataset(data.Dataset):
    def __init__(self, img_dir, has_label, transform=None, sort=False):
        olddir = os.getcwd()
        os.chdir(img_dir)
        classes = []
        
        if has_label:
            filenames = glob.glob('*/*.jpg')
        else:
            filenames = glob.glob('*.jpg')
        
        if sort:
            filenames = sorted(filenames)

        if has_label:
            #images = [None]*num
            for i, fn in enumerate(filenames):
                #images[i] = pil_load(os.path.join(DATA_DIR, fn))
                img_cls = int(fn.split('/')[-2].split('_')[-1]) - 1
                classes.append(img_cls)

        os.chdir(olddir)

        self.img_dir = img_dir
        self.transform = transform
        self.num       = len(filenames)
        self.filenames = filenames
        #self.images    = images
        self.has_label = has_label

        #print(self.img_dir)
        #print(self.num)
        
        if has_label:
            self.labels  = classes

    def __getitem__(self, index):
        fn = self.filenames[index]
        img = pil_load(os.path.join(self.img_dir, fn))
        #img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.has_label:
            label = self.labels[index]
            #return img, label, index
            return img, label, self.filenames[index]
        else:
            return img, self.filenames[index]

    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return self.num

data_transforms = {
    'train': transforms.Compose([
        #transforms.Scale(320), # for vgg19
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
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

def get_train_loader(batch_size = 16, shuffle = True):
    img_dir = TRAIN_DIR
    dset = CScreenDataset(img_dir, True, data_transforms['train'])
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
    dloader.num = dset.num
    return dloader

def get_val_loader(batch_size = 16, shuffle = True):
    img_dir = VAL_DIR
    dset = CScreenDataset(img_dir, True, data_transforms['valid'])
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
    dloader.num = dset.num
    dloader.filenames = dset.filenames
    return dloader

def get_stage1_test_loader(modelname, batch_size=16):
    img_dir = settings.STATGE1_TEST_DATA_PATH
    if modelname == 'inceptionv3':
        transkey = 'testv3'
    else:
        transkey = 'test'
    dset = CScreenDataset(img_dir, False, data_transforms[transkey], sort=True)
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
    dloader.num = dset.num
    dloader.filenames = dset.filenames
    return dloader

def get_stage2_test_loader(modelname, batch_size=16):
    try:
        img_dir = settings.STATGE2_TEST_DATA_PATH
    except:
        print('Please configure STATGE2_TEST_DATA_PATH in settings.py')
        exit()
    if img_dir is None or not os.path.exists(img_dir):
        print('Please configure STATGE2_TEST_DATA_PATH in settings.py correctly')
        exit()

    if modelname == 'inceptionv3':
        transkey = 'testv3'
    else:
        transkey = 'test'
    dset = CScreenDataset(img_dir, False, data_transforms[transkey], sort=True)
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
    dloader.num = dset.num
    dloader.filenames = dset.filenames
    return dloader

if __name__ == '__main__':
    loader = get_stage1_test_loader('res50')
    print(loader.num)
    for i, data in enumerate(loader):
        if i == 0:
            img, fn = data
            print(fn)