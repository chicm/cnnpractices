import cfg
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
from sklearn.metrics import log_loss


#data_dir = '/home/chicm/ml/cnnpractices/cervc/data/crop'
data_dir = cfg.INPUT_DIR + '/resize640'
#data_dir = '/home/chicm/ml/cnnpractices/cervc/data/orig400'
MODEL_DIR = data_dir + '/models'
RESULT_DIR = data_dir + '/results'
PRED_FILE = RESULT_DIR + '/pred_ens_torch.dat'
batch_size = 16
epochs = 100

def randomRotate(img):
    d = random.randint(0,4) * 45
    img2 = img.rotate(d, resample=Image.NEAREST)
    return img2
    
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        #transforms.Scale(256),
        #transforms.CenterCrop(224),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.Lambda(lambda x: randomFlip(x)),
        #transforms.Lambda(lambda x: randomTranspose(x)),
        #transforms.Lambda(lambda x: randomRotate(x)),
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
    ])
}

dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'valid', 'test']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'valid']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'valid']}
dset_classes = dsets['train'].classes
dsets['test'].imgs = sorted(dsets['test'].imgs)
test_loader = torch.utils.data.DataLoader(dsets['test'], batch_size=batch_size,
                                               shuffle=False, num_workers=4)
val_loader = torch.utils.data.DataLoader(dsets['valid'], batch_size=batch_size,
                                               shuffle=False, num_workers=4)

use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, w_file, lr_scheduler, init_lr=0.001, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0

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
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

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

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
                torch.save(best_model.state_dict(), w_file)

        print('epoch {}: {:.0f}s'.format(epoch, time.time()-epoch_since))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.6**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer    

def lr_scheduler2(optimizer, epoch, init_lr=0.001, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5**(epoch // lr_decay_epoch))
    if lr < 1e-6:
        lr = 0.0001

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer    

def create_res50():
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.cuda()
    return model_ft

def create_res101():
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.cuda()
    return model_ft

def create_res152():
    res152 = models.resnet152(pretrained=True)
    num_ftrs = res152.fc.in_features
    res152.fc = nn.Linear(num_ftrs, 3)
    res152 = res152.cuda()
    return res152

def create_inceptionv3():
    incept_ft = models.inception_v3(pretrained=True)
    num_ftrs = incept_ft.fc.in_features
    incept_ft.fc = nn.Linear(num_ftrs, 3)
    incept_ft.aux_logits=False
    print(num_ftrs)
    incept_ft = incept_ft.cuda()
    return incept_ft

def create_dense161():
    desnet_ft = models.densenet161(pretrained=True)
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Linear(num_ftrs, 3)
    print(num_ftrs)
    desnet_ft = desnet_ft.cuda()
    return desnet_ft

def create_dense169():
    desnet_ft = models.densenet169(pretrained=True)
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Linear(num_ftrs, 3)
    print(num_ftrs)
    desnet_ft = desnet_ft.cuda()
    return desnet_ft

def create_dense201():
    desnet_ft = models.densenet201(pretrained=True)
    num_ftrs = desnet_ft.classifier.in_features
    desnet_ft.classifier = nn.Linear(num_ftrs, 3)
    print(num_ftrs)
    desnet_ft = desnet_ft.cuda()
    return desnet_ft

def train(model, w_file):
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = train_model(model, criterion, optimizer_ft, w_file, lr_scheduler, init_lr=0.001, 
                        num_epochs=epochs)
    return model

def create_models():
    models = []
    models.append(create_dense201())
    models.append(create_dense161())
    models.append(create_res101())
    models.append(create_res152())
    return models

def train_inception_v3():
    print('training inception_v3')
    model = create_inceptionv3()
    w_file = MODEL_DIR + '/inceptionv3.pth'
    try:
        model.load_state_dict(torch.load(w_file))
    except:
        print('{} not found, continue'.format(w_file))
        pass
    train(model, w_file+'.new')

def train_res50():
    print('training resnet 50')
    model = create_res50()
    w_file = MODEL_DIR + '/res50.pth'
    try:
        model.load_state_dict(torch.load(w_file))
    except:
        print('{} not found, continue'.format(w_file))
        pass
    train(model, w_file+'.new')


def train_res101():
    print('training resnet 101')
    model = create_res101()
    w_file = MODEL_DIR + '/res101.pth'
    try:
        model.load_state_dict(torch.load(w_file))
    except:
        print('{} not found, continue'.format(w_file))
        pass
    train(model, w_file+'.new')


def train_res152():
    print('training resnet 152')
    model = create_res152()
    w_file = MODEL_DIR + '/res152.pth'
    try:
        model.load_state_dict(torch.load(w_file))
    except:
        print('{} not found, continue'.format(w_file))
        pass
    train(model, w_file+'.new')


def train_dense161():
    print('training densenet 161')
    model = create_dense161()
    w_file = MODEL_DIR + '/dense161.pth'
    try:
        model.load_state_dict(torch.load(w_file))
    except:
        print('{} not found, continue'.format(w_file))
        pass
    train(model, w_file+'.new')

def train_dense169():
    print('training densenet 169')
    model = create_dense169()
    w_file = MODEL_DIR + '/dense169.pth'
    try:
        model.load_state_dict(torch.load(w_file))
    except:
        print('{} not found, continue'.format(w_file))
        pass
    train(model, w_file+'.new')

def train_dense201():
    print('training densenet 201')
    model = create_dense201()
    w_file = MODEL_DIR + '/dense201.pth'
    try:
        model.load_state_dict(torch.load(w_file))
    except:
        print('{} not found, continue'.format(w_file))
        pass
    train(model, w_file+'.new')

def train_all():
    train_dense201()
    train_dense161()
    train_dense169()
    train_res101()
    train_res152()
    train_res50()

def make_preds(net, test_loader):
    preds = []
    m = nn.Softmax()
    for i, (img, indices) in enumerate(test_loader, 0):
        #print(i, img.size())
        inputs = Variable(img.cuda())
        outputs = net(inputs)
        pred = m(outputs).data.cpu().tolist()
        #print(pred[:1])
        preds.append(pred)
        #for p in pred:
        #    preds.append(p)
    return preds

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def log_loss_func(preds, weights, y):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros(preds[0].shape)
    for weight, prediction in zip(weights, preds):
            final_prediction += weight*prediction
    print(y[:5])
    print(final_prediction[:5])
    print(y.shape)
    print(final_prediction.shape)
    loss = log_loss(y, final_prediction)
    print(loss)
    return loss

def find_weights():
    dense169 = create_dense169()
    dense169.load_state_dict(torch.load(MODEL_DIR + '/dense169.pth'))
    res152 = create_res152()
    res152.load_state_dict(torch.load(MODEL_DIR + '/res152.pth'))
    dense201 = create_dense201()
    dense201.load_state_dict(torch.load(MODEL_DIR + '/dense201.pth'))
    dense161 = create_dense161()
    dense161.load_state_dict(torch.load(MODEL_DIR + '/dense161.pth'))

    
    #pred1 = make_preds(res101, test_loader)
    #pred1 = np.array(pred1).reshape(512,3)
    pred1 = make_preds(dense169, val_loader)
    #print(pred1)
    print(np.array(pred1).shape)
    pred1 = np.array(pred1).reshape(600,3)
    pred2 = make_preds(res152, val_loader)
    pred2 = np.array(pred2).reshape(600,3)
    pred3 = make_preds(dense201, val_loader)
    pred3 = np.array(pred3).reshape(600,3)
    pred4 = make_preds(dense161, val_loader)
    pred4 = np.array(pred4).reshape(600,3)

    y = []
    for data, labels in val_loader:
        for label in labels:
            y.append(label)
    print(np.array(y).shape)
    print(y[:10])
 
    log_loss_func([pred1,pred2, pred3, pred4], [0.25, 0.25, 0.25, 0.25], np.array(one_hot(y)))

def encode(x):
    if x == 0:
        return [1, 0, 0]
    elif x == 1:
        return [0, 1, 0]
    elif x == 2 :
        return [0, 0, 1]

def one_hot(label):
    return [encode(x) for x in label]

def ensemble():
    res101 = create_res101()
    res101.load_state_dict(torch.load(MODEL_DIR + '/res101.pth'))
    #dense169 = create_dense169()
    #dense169.load_state_dict(torch.load(MODEL_DIR + '/dense169.pth'))
    res152 = create_res152()
    res152.load_state_dict(torch.load(MODEL_DIR + '/res152.pth'))
    dense201 = create_dense201()
    dense201.load_state_dict(torch.load(MODEL_DIR + '/dense201.pth'))
    dense161 = create_dense161()
    dense161.load_state_dict(torch.load(MODEL_DIR + '/dense161.pth'))


    pred1 = make_preds(res101, test_loader)
    pred1 = np.array(pred1).reshape(512,3)
    #pred1 = make_preds(dense169, test_loader)
    #pred1 = np.array(pred1).reshape(512, 3)
    pred2 = make_preds(res152, test_loader)
    pred2 = np.array(pred2).reshape(512,3)
    pred3 = make_preds(dense201, test_loader)
    pred3 = np.array(pred3).reshape(512,3)
    pred4 = make_preds(dense161, test_loader)
    pred4 = np.array(pred4).reshape(512,3)

    preds = np.mean([pred1, pred2, pred3, pred4], axis=0)
    save_array(PRED_FILE, preds)
    print(preds[:10])

def do_clip(arr, mx): 
    return np.clip(arr, (1-mx)/2, mx)

def submit(filename, clip):
    filenames = [f.split('/')[-1] for f, i in dsets['test'].imgs]

    preds = load_array(PRED_FILE)
    subm = do_clip(preds, clip)
    classes = dsets['train'].classes
    subm_name = RESULT_DIR+'/'+filename  
    submission = pd.DataFrame(subm, columns=classes)
    submission.insert(0, 'image_name', filenames)
    print(submission.head())
    submission.to_csv(subm_name, index=False)

parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs=1, help="train model")
parser.add_argument("--fw", action='store_true', help="find weights")
parser.add_argument("--ens", action='store_true', help="ensemble predict")
parser.add_argument("--sub", nargs=2, help="generate submission file")

args = parser.parse_args()
if args.train:
    print('start training model')
    mname = args.train[0]
    if mname == 'dense201':
        train_dense201()
    elif mname == 'dense161':
        train_dense161()
    elif mname == 'dense169':
        train_dense169()
    elif mname == 'res50':
        train_res50()
    elif mname == 'res101':
        train_res101()
    elif mname == 'res152':
        train_res152()
    elif mname == 'inceptionv3':
        train_inception_v3()
    elif mname == 'all':
        train_all()
    else:
        print('model name {} not found'.format(mname))
    print('done')
if args.ens:
    ensemble()
    print('done')
if args.fw:
    find_weights()
if args.sub:
    print('generating submision file...')
    submit(args.sub[0], (float)(args.sub[1]))
    print('done')
