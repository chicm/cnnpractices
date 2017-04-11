
import os, shutil, glob, sys
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from utils import *
#from vgg16 import *
from keras import applications
import argparse

DATA_DIR = '/home/chicm/ml/cnnpractices/statefarm/data'
TRAIN_DIR = DATA_DIR+'/train'
TEST_DIR = DATA_DIR + '/test'
VALID_DIR = DATA_DIR + '/valid'
RESULT_DIR = DATA_DIR + '/results'

TRAIN_FEAT = RESULT_DIR + '/train_feat2.dat'
VAL_FEAT = RESULT_DIR + '/val_feat2.dat'
TEST_FEAT = RESULT_DIR + '/test_feat2.dat'
WEIGHTS_FILE = RESULT_DIR + '/sf_weights2.h5'
PREDICTS_FILE = RESULT_DIR + '/predicts2'

batch_size = 32

def do_clip(arr, mx): 
    return np.clip(arr, (1-mx)/9, mx)

def move_validation_back():
    os.chdir(VALID_DIR)
    files = glob('*/*.jpg')
    for fn in files:
        shutil.move(VALID_DIR+'/'+fn, TRAIN_DIR+'/'+fn)

def create_validation_data():
    drivers_ds = pd.read_csv(DATA_DIR + '/driver_imgs_list.csv')
    drivers_ds.head
    img2driver = drivers_ds.set_index('img')['subject'].to_dict()
    driver2imgs = {k: g["img"].tolist() 
                for k, g in drivers_ds[['subject', 'img']].groupby("subject")}

    #drivers = sorted(driver2imgs.keys())
    drivers = np.random.permutation(driver2imgs.keys())
    print drivers

    for i in range(4):
        filenames = driver2imgs.get(drivers[i])
        for fn in filenames:
            cls = drivers_ds.set_index('img').classname.get(fn)
            shutil.move(TRAIN_DIR+'/'+cls+'/'+fn, VALID_DIR+'/'+cls+'/'+fn)

#def get_vgg_model():
#    vgg = Vgg16()
#    model = vgg.model
#    return model
def get_keras_vgg_model():
    model = applications.VGG16(include_top=True, weights='imagenet')
    return model

def gen_vgg_features(gen_train=False, gen_valid=False, gen_test=False):
    gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05,
		shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)

    da_batches = get_batches(TRAIN_DIR, gen_t,  batch_size = batch_size, shuffle=False)
    val_batches = get_batches(VALID_DIR, batch_size = batch_size, shuffle=False)
    test_batches = get_batches(TEST_DIR, batch_size = batch_size, shuffle=False)

    model = get_keras_vgg_model()

    last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D] [-1]
    conv_layers = model.layers[:last_conv_idx+1]

    conv_model = Sequential(conv_layers)

    if gen_train:
        da_conv_feat = conv_model.predict_generator(da_batches, da_batches.nb_sample*3)
        save_array(TRAIN_FEAT, da_conv_feat)
    if gen_valid:
        conv_val_feat = conv_model.predict_generator(val_batches, val_batches.nb_sample)
        save_array(VAL_FEAT, conv_val_feat)
    if gen_test:
        conv_test_feat = conv_model.predict_generator(test_batches, test_batches.nb_sample)
        save_array(TEST_FEAT, conv_test_feat)

def get_conv_layers(model):
    last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D] [-1]
    conv_layers = model.layers[:last_conv_idx+1]
    return conv_layers

def show_conv():
    model = get_keras_vgg_model()
    print model.summary()
    conv_layers = get_conv_layers(model)
    conv_model = Sequential(conv_layers)
    #print conv_model.summary()

def get_bn_layers(p):
    model = get_keras_vgg_model()
    last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D] [-1]
    conv_layers = model.layers[:last_conv_idx+1]

    return [
        MaxPooling2D(input_shape = conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dropout(p/2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
    ]

def get_bn_model():
    p = 0.8
    bn_model = Sequential(get_bn_layers(p))
    bn_model.compile(Adam(lr=0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])
    return bn_model

def train_bn_layers():
    conv_val_feat = load_array(VAL_FEAT)
    da_conv_feat = load_array(TRAIN_FEAT)
    (val_classes, trn_classes, val_labels, trn_labels, val_filenames, trn_filenames, test_filenames) = get_classes(DATA_DIR+'/')

    da_trn_labels = np.concatenate([trn_labels]*3)

    bn_model = get_bn_model()
    
    bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=2, 
             validation_data=(conv_val_feat, val_labels))

    bn_model.optimizer.lr = 0.01
    bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=8, 
                validation_data=(conv_val_feat, val_labels))

    bn_model.optimizer.lr = 0.0001
    bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=10, 
                validation_data=(conv_val_feat, val_labels))

    bn_model.save_weights(WEIGHTS_FILE)

def save_predict():
    bn_model = get_bn_model()
    bn_model.load_weights(WEIGHTS_FILE)
    conv_test_feat = load_array(TEST_FEAT)
    preds = bn_model.predict(conv_test_feat, batch_size=batch_size)
    save_array(PREDICTS_FILE, preds)

def gen_submit(submit_filename, clip_percentage):
    preds = load_array(PREDICTS_FILE)

    subm = do_clip(preds, clip_percentage)
    subm_name = DATA_DIR+'/results/' + submit_filename

    trn_batches = get_batches(DATA_DIR+'/train', batch_size = batch_size)
    classes = sorted(trn_batches.class_indices, key=trn_batches.class_indices.get)
    #print classes
    batches = get_batches(DATA_DIR+'/test', batch_size = batch_size, shuffle=False)

    submission = pd.DataFrame(subm, columns=classes)
    submission.insert(0, 'img', [a[8:] for a in batches.filenames])
    print submission.head()
    submission.to_csv(subm_name, index=False, compression='gzip')


parser = argparse.ArgumentParser()
parser.add_argument("--mb", action='store_true', help="move validation data back to training directory")
parser.add_argument("--createval", action='store_true',
                    help="create validation data from training data")
parser.add_argument("--gentestfeats", action='store_true',
                    help="generate vgg conv layers output array for test data")
parser.add_argument("--gentrainfeats", action='store_true',
                    help="generate vgg conv layers output array for train and validation data")
parser.add_argument("--train", action='store_true', help="train dense layers")
parser.add_argument("--predict", action='store_true', help="predict test data and save")
parser.add_argument("--sub", nargs=2, help="generate submission file")
parser.add_argument("--showconv", action='store_true', help="show summary of conv model")

args = parser.parse_args()
if args.mb:
    print 'moving back...'
    move_validation_back()
    print 'done'
if args.createval:
    print 'creating val data...'
    create_validation_data()
    print 'done'
if args.gentestfeats:
    print 'generating test features...'
    gen_vgg_features(gen_test=True)
    print 'done'
if args.gentrainfeats:
    print 'generating train and val features...'
    gen_vgg_features(gen_train=True, gen_valid=True)
    print 'done'
if args.train:
    print 'training dense layer...'
    train_bn_layers()
    print 'done'
if args.predict:
    print 'predicting test data...'
    save_predict()
    print 'done'
if args.sub:
    print 'generating submision file...'
    gen_submit(args.sub[0], (float)(args.sub[1]))
    print 'done'
if args.showconv:
    show_conv()
