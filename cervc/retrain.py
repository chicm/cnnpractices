
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
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from vgg16bn import *
from keras import applications
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import random

DATA_DIR = '/home/chicm/data/cervc/clean640'
TRAIN_DIR = DATA_DIR+'/train'
TEST_DIR = DATA_DIR + '/test2'
VALID_DIR = DATA_DIR + '/valid'
RESULT_DIR = DATA_DIR + '/results2'

TRAIN_FEAT = RESULT_DIR + '/train_feat.dat'
DA_TRAIN_FEAT = RESULT_DIR + '/da_train_feat.dat'
VAL_FEAT = RESULT_DIR + '/val_feat.dat'
TEST_FEAT = RESULT_DIR + '/test_feat.dat'
WEIGHTS_FILE = RESULT_DIR + '/sf_weights.h5'
PREDICTS_FILE = RESULT_DIR + '/predicts'

batch_size = 64
da_multi = 10

def do_clip(arr, mx): 
    return np.clip(arr, (1-mx)/2, mx)


def remove_last_conv(model):
    while model.layers[-1].name != 'convolution2d_12':
        model.pop()

def get_my_vgg_model():
    vgg = Vgg16BN()
    model = vgg.model
    remove_last_conv(model)
    return model

def get_keras_vgg_model():
    model = applications.VGG16(include_top=True, weights='imagenet')
    return model

def get_vgg_model():
    return get_my_vgg_model()

def get_bn_layers(input_shape=(512,14,14)):
    return [
        ZeroPadding2D(input_shape=input_shape),
        Convolution2D(512, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.6),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.8),
        Dense(3, activation='softmax')
    ]

def get_bn_model():
    bn_model = Sequential(get_bn_layers())
    bn_model.compile(Adam(lr=0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])
    return bn_model


def gen_vgg_features(gen_train=False, gen_valid=False, gen_test=False):
    gen_t = image.ImageDataGenerator(height_shift_range=0.05,
		shear_range=0.1, channel_shift_range=20, width_shift_range=0.1, horizontal_flip=True, 
        vertical_flip=True)

    da_batches = get_batches(TRAIN_DIR, gen_t,  batch_size = batch_size, shuffle=False)
    batches = get_batches(TRAIN_DIR, batch_size = batch_size, shuffle=False)
    val_batches = get_batches(VALID_DIR, batch_size = batch_size, shuffle=False)
    test_batches = get_batches(TEST_DIR, batch_size = batch_size, shuffle=False)

    model = get_vgg_model()

    last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D] [-1]
    conv_layers = model.layers[:last_conv_idx+1]

    conv_model = Sequential(conv_layers)

    if gen_train:
        da_conv_feat = conv_model.predict_generator(da_batches, da_batches.nb_sample*da_multi)
        save_array(DA_TRAIN_FEAT, da_conv_feat)
        conv_feat = conv_model.predict_generator(batches, batches.nb_sample)
        save_array(TRAIN_FEAT, conv_feat)
    if gen_valid:
        conv_val_feat = conv_model.predict_generator(val_batches, val_batches.nb_sample)
        save_array(VAL_FEAT, conv_val_feat)
    if gen_test:
        conv_test_feat = conv_model.predict_generator(test_batches, test_batches.nb_sample)
        save_array(TEST_FEAT, conv_test_feat)

def show_conv():
    model = get_vgg_model()
    #remove_last_conv(model)
    print model.summary()
    print model.layers[-1].output_shape
    #last_index = -1
    
    #for index, layer in enumerate(model.layers):
    #    print layer.name, layer.output_shape
    #    if layer.name == 'zeropadding2d_11':
    #        last_index = index
    #print last_index 
    #print dir(model.layers[0])
    #print model.summary()
    #bn_model = get_bn_model()
    #print bn_model.summary()
    #conv_layers = get_conv_layers(model)
    #conv_model = Sequential(conv_layers)
    #print conv_model.summary()


def train_bn_layers():
    conv_val_feat = load_array(VAL_FEAT)
    print conv_val_feat.shape
    da_conv_feat = load_array(DA_TRAIN_FEAT)
    conv_feat = load_array(TRAIN_FEAT)
    da_conv_feat = np.concatenate([da_conv_feat, conv_feat])
    print da_conv_feat.shape

    (val_classes, trn_classes, val_labels, trn_labels, val_filenames, trn_filenames, test_filenames) = get_classes(DATA_DIR+'/')

    da_trn_labels = np.concatenate([trn_labels]*(da_multi+1))
    print da_trn_labels.shape

    for i in range(5):
        bn_model = get_bn_model()
        
        bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=10, 
                validation_data=(conv_val_feat, val_labels))

        K.set_value(bn_model.optimizer.lr, 0.01)
        #bn_model.optimizer.lr = 0.01
        bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=10, 
                    validation_data=(conv_val_feat, val_labels))

        #bn_model.optimizer.lr = 0.0001
        #bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=10, 
        #            validation_data=(conv_val_feat, val_labels))
        kfold_weights_path = WEIGHTS_FILE+str(random.random())
        callbacks = [
                EarlyStopping(monitor='val_loss', patience=40, verbose=0),
                ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

        #bn_model.optimizer.lr = 0.00001
        K.set_value(bn_model.optimizer.lr, 0.00001)
        bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=300, 
                    validation_data=(conv_val_feat, val_labels), callbacks=callbacks)

    #bn_model.save_weights(WEIGHTS_FILE+str(random.random()))

def ensemble():
    preds = []
    test_feat = load_array(TEST_FEAT)
    w_files = glob(WEIGHTS_FILE+'*')
    for fn in w_files:
        model = get_bn_model()
        #fn = '{:s}/w{:d}.h5'.format(RESULT_DIR, i+1)
        print fn
        model.load_weights(fn)
        preds.append(model.predict(test_feat, batch_size=batch_size))
    print np.array(preds).shape
    m = np.mean(preds, axis=0)
    save_array(PREDICTS_FILE, m)
        

def save_predict():
    bn_model = get_bn_model()
    bn_model.load_weights(WEIGHTS_FILE)
    conv_test_feat = load_array(TEST_FEAT)
    preds = bn_model.predict(conv_test_feat, batch_size=batch_size)
    save_array(PREDICTS_FILE, preds)

def gen_submit(submit_filename, clip_percentage):
    preds = load_array(PREDICTS_FILE)
    print preds[:20]
    subm = do_clip(preds, clip_percentage)
    subm_name = RESULT_DIR+'/' + submit_filename

    trn_batches = get_batches(DATA_DIR+'/train', batch_size = batch_size)
    classes = sorted(trn_batches.class_indices, key=trn_batches.class_indices.get)
    #print classes
    batches = get_batches(DATA_DIR+'/test', batch_size = batch_size, shuffle=False)

    submission = pd.DataFrame(subm, columns=classes)
    submission.insert(0, 'image_name', [a[8:] for a in batches.filenames])
    #print [a for a in batches.filenames][:10]
    print submission.head()
    submission.to_csv(subm_name, index=False)


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
parser.add_argument("--ens", action='store_true', help="ensemble predict")

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
if args.ens:
    print 'ensemble predicting...'
    ensemble()
    print 'done'
if args.sub:
    print 'generating submision file...'
    gen_submit(args.sub[0], (float)(args.sub[1]))
    print 'done'
if args.showconv:
    show_conv()
