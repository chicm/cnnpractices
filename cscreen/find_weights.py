import settings
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from utils import save_array, load_array, create_dense161, create_dense201, create_res101, create_res152

data_dir = settings.DATA_DIR
RESULT_DIR = settings.DATA_DIR + '/results'
PRED_FILE_RAW = RESULT_DIR + '/pred_ens_raw.dat'
batch_size = 16

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
dsets['valid'].imgs = sorted(dsets['valid'].imgs)
val_loader = torch.utils.data.DataLoader(dsets['valid'], batch_size=batch_size,
                                               shuffle=False, num_workers=4)

#print(dsets['valid'].imgs)
def encode(x):
    if x == 0:
        return [1, 0, 0]
    elif x == 1:
        return [0, 1, 0]
    elif x == 2 :
        return [0, 0, 1]

def one_hot(label):
    return [encode(x) for x in label]

def make_preds(net, data_loader):
    preds = []
    y = []
    m = nn.Softmax()
    #data_loader.
    for i, (img, indices) in enumerate(data_loader, 0):
        #print(i, img.size())
        inputs = Variable(img.cuda())
        outputs = net(inputs)
        pred = m(outputs).data.cpu().tolist()
        for p in pred:
            preds.append(p)
        #print(indices)
        for t in indices.tolist():
            y.append(t)
        #print(pred)
        #print(indices.tolist())
        print(log_loss(np.array(one_hot(indices.tolist())), np.array(pred)))
    return np.array(preds), np.array(y)



def find_weights():
    res101,_ = create_res101(True)
    res152,_ = create_res152(True)
    dense201,_ = create_dense201(True)
    dense161,_ = create_dense161(True)
    
    
    pred1, y1 = make_preds(res101, val_loader)
    #print(y1)
    #pred1 = np.random.random((600, 3))
    #print(pred1[:5])
    pred2, y2 = make_preds(res152, val_loader)
    #print(y2)
    pred3, y3 = make_preds(dense201, val_loader)
    pred4, y4 = make_preds(dense161, val_loader)

    #_, preds_test = torch.max(torch.from_numpy(pred4), 1)
    #print(preds_test[:10])
    print(pred1.shape)
    #print(log_loss(y1, y2))
    #print(log_loss(y3, y2))
    #print(log_loss(y3, y4))
    
    y = []
    for data, labels in val_loader:
        for label in labels:
            y.append(label)
    print(np.array(y).shape)
    #print(y)

    return [pred1,pred2, pred3, pred4], y
    #print(y)

    #loss1 = log_loss(y, pred1)
    #print(loss1)
    #loss2 = log_loss(y, pred2)
    #print(loss2)
    #loss3 = log_loss(y, pred3)
    #print(loss3)
    #loss4 = log_loss(y, pred4)
    #print(loss4)
 
    #log_loss_func([pred1,pred2, pred3, pred4], [0.25, 0.25, 0.25, 0.25], np.array(one_hot(y)))

preds, y = find_weights()
y = np.array(one_hot(y))
print(preds[0].shape)
print(y.shape)

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros(preds[0].shape)
    for weight, prediction in zip(weights, preds):
            final_prediction += weight*prediction
    #print(y[:5])
    #print(final_prediction[:5])
    #print(y.shape)
    #print(final_prediction.shape)
    loss = log_loss(y, final_prediction)
    #print(loss)
    return loss

#the algorithms need a starting value, right not we chose 0.5 for all weights
#its better to choose many random starting points and run minimize a few times
starting_values = [0.1]*len(preds)

#adding constraints  and a different solver as suggested by user 16universe
#https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(preds)

res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))