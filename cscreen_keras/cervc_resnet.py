
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
#from vgg16bn import *
#from keras import applications
from resnet50 import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

DATA_DIR = '/home/chicm/ml/cnnpractices/cervc/data/full640'
TRAIN_DIR = DATA_DIR+'/train'
TEST_DIR = '/home/chicm/ml/cnnpractices/cervc/data/full/test'
VALID_DIR = DATA_DIR + '/valid'
RESULT_DIR = DATA_DIR + '/results'

TRAIN_FEAT = RESULT_DIR + '/train_feat.dat'
DA_TRAIN_FEAT = RESULT_DIR + '/da_train_feat.dat'
VAL_FEAT = RESULT_DIR + '/val_feat.dat'
TEST_FEAT = RESULT_DIR + '/test_feat.dat'
WEIGHTS_FILE = RESULT_DIR + '/sf_weights.h5'
PREDICTS_FILE = RESULT_DIR + '/predicts'

batch_size = 16
#da_multi = 1
img_size = (640,640)

def do_clip(arr, mx): 
    return np.clip(arr, (1-mx)/2, mx)

def move_validation_back():
    os.chdir(VALID_DIR)
    files = glob('*/*.jpg')
    for fn in files:
        shutil.move(VALID_DIR+'/'+fn, TRAIN_DIR+'/'+fn)

def create_validation_data():
    os.chdir(TRAIN_DIR)
    files = glob('*/*.jpg')

    #drivers = sorted(driver2imgs.keys())
    files = np.random.permutation(files)
    print files[:10]

    for i in range(600):
        fn = files[i]
        #print TRAIN_DIR+'/'+fn
        shutil.move(TRAIN_DIR+'/'+fn, VALID_DIR+'/'+fn)

def get_res_model():
    resnet = Resnet50(size=(640,640), include_top=False)
    return resnet.model

def gen_res_features(gen_train=False, gen_valid=False, gen_test=False):
    gen_t = image.ImageDataGenerator(rotation_range=180, height_shift_range=0.1,
		shear_range=0.1, channel_shift_range=20, width_shift_range=0.1, horizontal_flip=True, 
        vertical_flip=True)

    #da_batches = get_batches(TRAIN_DIR, gen_t,  batch_size = batch_size, shuffle=False, target_size=img_size)
    batches = get_batches(TRAIN_DIR, batch_size = batch_size, shuffle=False, target_size=img_size)
    val_batches = get_batches(VALID_DIR, batch_size = batch_size, shuffle=False, target_size=img_size)
    test_batches = get_batches(TEST_DIR, batch_size = batch_size, shuffle=False, target_size=img_size)

    model = get_res_model()

    if gen_train:
        #da_conv_feat = conv_model.predict_generator(da_batches, da_batches.nb_sample*da_multi)
        #save_array(DA_TRAIN_FEAT, da_conv_feat)
        conv_feat = model.predict_generator(batches, batches.nb_sample)
        save_array(TRAIN_FEAT, conv_feat)
    if gen_valid:
        conv_val_feat = model.predict_generator(val_batches, val_batches.nb_sample)
        save_array(VAL_FEAT, conv_val_feat)
    if gen_test:
        conv_test_feat = model.predict_generator(test_batches, test_batches.nb_sample)
        save_array(TEST_FEAT, conv_test_feat)

def show_conv():
    model = get_res_model()
    print model.summary()
    lrn_model = get_lrn_model()
    print lrn_model.summary()
    #conv_layers = get_conv_layers(model)
    #conv_model = Sequential(conv_layers)
    #print conv_model.summary()

def get_lrn_layers():
    conv_layers = get_res_model().layers
    nf=128 
    p=0.8
    return [
        BatchNormalization(axis=1, input_shape=conv_layers[-1].output_shape[1:]),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D((1,2)),
        Convolution2D(3,3,3, border_mode='same'),
        Dropout(p),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ]

def get_lrn_model():
    lrn_model = Sequential(get_lrn_layers())
    lrn_model.compile(Adam(lr=0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])
    return lrn_model

def train_lrn_layers():
    conv_val_feat = load_array(VAL_FEAT)
    print conv_val_feat.shape
    #da_conv_feat = load_array(DA_TRAIN_FEAT)
    conv_feat = load_array(TRAIN_FEAT)
    #da_conv_feat = np.concatenate([da_conv_feat, conv_feat])
    print conv_feat.shape

    (val_classes, trn_classes, val_labels, trn_labels, val_filenames, trn_filenames, test_filenames) = get_classes(DATA_DIR+'/')

    #da_trn_labels = np.concatenate([trn_labels]*(da_multi))
    #da_trn_labels = trn_labels
    print trn_labels.shape

    lrn_model = get_lrn_model()
    batch_size = 128
    
    lrn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=4, 
             validation_data=(conv_val_feat, val_labels))

    lrn_model.optimizer.lr = 0.01
    lrn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=4, 
                validation_data=(conv_val_feat, val_labels))

    lrn_model.optimizer.lr = 0.0001
    lrn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=10, 
                validation_data=(conv_val_feat, val_labels))

    lrn_model.save_weights(WEIGHTS_FILE+'_stg1')
    
    lrn_model.optimizer.lr = 0.00001
    lrn_model.fit(conv_feat, trn_labels, batch_size=batch_size, nb_epoch=10, 
                validation_data=(conv_val_feat, val_labels))

    lrn_model.save_weights(WEIGHTS_FILE)

def save_predict():
    lrn_model = get_lrn_model()
    lrn_model.load_weights(WEIGHTS_FILE)
    conv_test_feat = load_array(TEST_FEAT)
    preds = lrn_model.predict(conv_test_feat, batch_size=batch_size)
    save_array(PREDICTS_FILE, preds)

def gen_submit(submit_filename, clip_percentage):
    preds = load_array(PREDICTS_FILE)
    print preds[:20]
    subm = do_clip(preds, clip_percentage)
    subm_name = RESULT_DIR + '/' + submit_filename

    trn_batches = get_batches(DATA_DIR+'/train', batch_size = batch_size)
    classes = sorted(trn_batches.class_indices, key=trn_batches.class_indices.get)
    #print classes
    batches = get_batches(TEST_DIR, batch_size = batch_size, shuffle=False)

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
    gen_res_features(gen_test=True)
    print 'done'
if args.gentrainfeats:
    print 'generating train and val features...'
    gen_res_features(gen_train=True, gen_valid=True)
    print 'done'
if args.train:
    print 'training dense layer...'
    train_lrn_layers()
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
