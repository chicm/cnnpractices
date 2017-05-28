
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
from vgg16bn import *
from keras import applications
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

FIRST_DATA_DIR = '/home/chicm/ml/cnnpractices/cervc/data/first'
FIRST_TRAIN_DIR = '/home/chicm/ml/cnnpractices/cervc/data/first/train'
 

DATA_DIR = '/home/chicm/ml/cnnpractices/cervc/data/full'

TRAIN_DIR = DATA_DIR+'/train'
TEST_DIR = DATA_DIR + '/test'
VALID_DIR = DATA_DIR + '/valid'
RESULT_DIR = DATA_DIR + '/results'

TRAIN_FEAT = RESULT_DIR + '/train_feat.dat'
DA_TRAIN_FEAT = RESULT_DIR + '/da_train_feat.dat'
VAL_FEAT = RESULT_DIR + '/val_feat.dat'
TEST_FEAT = RESULT_DIR + '/test_feat.dat'
WEIGHTS_FILE = RESULT_DIR + '/sf_weights.h5'
PREDICTS_FILE = RESULT_DIR + '/predicts'

FIRST_DA_TRAIN_FEAT = RESULT_DIR + '/first_da_train_feat.dat'
FIRST_TRAIN_FEAT = RESULT_DIR + '/first_train_feat.dat'

DA_LABEL = RESULT_DIR + '/da_label.dat'
VAL_LABEL = RESULT_DIR + '/val_label.dat'
FIRST_LABEL = RESULT_DIR + '/first_label.dat'

batch_size = 32
da_multi = 5
da_multi_first = 15

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

def get_my_vgg_model():
    vgg = Vgg16BN()
    model = vgg.model
    return model
def get_keras_vgg_model():
    model = applications.VGG16(include_top=True, weights='imagenet')
    return model
def get_vgg_model():
    return get_my_vgg_model()

def gen_vgg_features(gen_train=False, gen_valid=False, gen_test=False, gen_first=False):
    gen_t = image.ImageDataGenerator(rotation_range=180, height_shift_range=0.1,
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

    if gen_first:
        first_batches = get_batches(FIRST_TRAIN_DIR, batch_size = batch_size, shuffle=False)
        first_da_batches = get_batches(FIRST_TRAIN_DIR, gen_t,  batch_size = batch_size, shuffle=False)

        first_da_conv_feat = conv_model.predict_generator(first_da_batches, first_da_batches.nb_sample*da_multi_first)
        save_array(FIRST_DA_TRAIN_FEAT, first_da_conv_feat)
        
        first_conv_feat = conv_model.predict_generator(first_batches, first_batches.nb_sample)
        save_array(FIRST_TRAIN_FEAT, first_conv_feat)

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

def get_conv_layers(model):
    last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D] [-1]
    conv_layers = model.layers[:last_conv_idx+1]
    return conv_layers

def show_conv():
    model = get_vgg_model()
    print model.summary()
    bn_model = get_bn_model()
    print bn_model.summary()
    #conv_layers = get_conv_layers(model)
    #conv_model = Sequential(conv_layers)
    #print conv_model.summary()

def get_bn_layers():
    model = get_vgg_model()
    last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D] [-1]
    conv_layers = model.layers[:last_conv_idx+1]

    return [
        MaxPooling2D(input_shape = conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.8),
        Dense(3, activation='softmax')
    ]

def get_bn_model():
    bn_model = Sequential(get_bn_layers())
    bn_model.compile(Adam(lr=0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])
    return bn_model

def train_bn_layers():
    conv_val_feat = bcolz.open(VAL_FEAT, mode='r')
    val_labels = bcolz.open(VAL_LABEL, mode='r')

    da_conv_feat = bcolz.open(DA_TRAIN_FEAT, mode='r')
    first_da_feat = bcolz.open(FIRST_DA_TRAIN_FEAT, mode='r')
    da_labels = bcolz.open(DA_LABEL, mode='r')
    first_labels = bcolz.open(FIRST_LABEL, mode='r')
    bs = da_conv_feat.chunklen*batch_size
    print bs
    batches = BcolzArrayIterator(da_conv_feat, da_labels, batch_size=bs, shuffle=False)
    first_batches = BcolzArrayIterator(first_da_feat, first_labels, batch_size=bs, shuffle=False)
    val_batches = BcolzArrayIterator(conv_val_feat, val_labels, batch_size=bs, shuffle=False)
    
    print val_batches.N
    print batches.N
    print first_batches.N
    
    new_batches = MixIterator([batches, first_batches])
    N = batches.N + first_batches.N
    print N

    model = get_bn_model()
    
    #bn_model.fit(da_conv_feat, da_trn_labels, batch_size=batch_size, nb_epoch=10, 
    #         validation_data=(conv_val_feat, val_labels))
    #for i in range(10):
    model.fit_generator(new_batches, samples_per_epoch=N, nb_epoch=10, validation_data=val_batches, nb_val_samples=val_batches.N)
        #model.fit_generator(first_batches, samples_per_epoch=first_batches.N, nb_epoch=1, validation_data=val_batches, nb_val_samples=val_batches.N)

    model.optimizer.lr = 0.01
    #for i in range(10):
    model.fit_generator(new_batches, samples_per_epoch=N, nb_epoch=10, validation_data=val_batches, nb_val_samples=val_batches.N)
        #model.fit_generator(first_batches, samples_per_epoch=first_batches.N, nb_epoch=1, validation_data=val_batches, nb_val_samples=val_batches.N)

    model.optimizer.lr = 0.00001
    #for i in range(30):
    model.fit_generator(batches, samples_per_epoch=N, nb_epoch=20, validation_data=val_batches, nb_val_samples=val_batches.N)
        #model.fit_generator(first_batches, samples_per_epoch=first_batches.N, nb_epoch=1, validation_data=val_batches, nb_val_samples=val_batches.N)
    model.save_weights(WEIGHTS_FILE)

def save_labels():
    first_batches = get_batches(FIRST_TRAIN_DIR, batch_size = batch_size, shuffle=False)
    first_labels = onehot(first_batches.classes)
    first_labels = np.concatenate([first_labels]*(da_multi_first)) 
    save_array(FIRST_LABEL, first_labels)

    (val_classes, trn_classes, val_labels, trn_labels, val_filenames, trn_filenames, test_filenames) = get_classes(DATA_DIR+'/')

    da_trn_labels = np.concatenate([trn_labels]*(da_multi))
    save_array(DA_LABEL, da_trn_labels)
    save_array(VAL_LABEL, val_labels)   

def train_bn_layers2():
    conv_val_feat = load_array(VAL_FEAT)
    print conv_val_feat.shape

    first_da_conv_feat = load_array(FIRST_DA_TRAIN_FEAT)
    first_conv_feat = load_array(FIRST_TRAIN_FEAT)
    first_da_conv_feat = np.concatenate([first_da_conv_feat, first_conv_feat])    
    
    print first_da_conv_feat.shape

    (val_classes, trn_classes, val_labels, trn_labels, val_filenames, trn_filenames, test_filenames) = get_classes(DATA_DIR+'/')

    first_batches = get_batches(FIRST_TRAIN_DIR, batch_size = batch_size, shuffle=False)
    first_labels = onehot(first_batches.classes)
    first_labels = np.concatenate([first_labels]*(da_multi_first+1))
    print first_labels.shape
    

    bn_model = get_bn_model()
    bn_model.load_weights(WEIGHTS_FILE)
    
    bn_model.fit(first_da_conv_feat, first_labels, batch_size=batch_size, nb_epoch=5, 
             validation_data=(conv_val_feat, val_labels))

    bn_model.optimizer.lr = 0.01
    bn_model.fit(first_da_conv_feat, first_labels, batch_size=batch_size, nb_epoch=5, 
             validation_data=(conv_val_feat, val_labels))

    bn_model.optimizer.lr = 0.00001
    bn_model.fit(first_da_conv_feat, first_labels, batch_size=batch_size, nb_epoch=10, 
             validation_data=(conv_val_feat, val_labels))

    bn_model.save_weights(WEIGHTS_FILE+'2')


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
    subm_name = DATA_DIR+'/results/' + submit_filename

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
parser.add_argument("--genfirstfeats", action='store_true',
                    help="generate vgg conv layers output array for first train data")
parser.add_argument("--gentrainfeats", action='store_true',
                    help="generate vgg conv layers output array for train and validation data")
parser.add_argument("--train", action='store_true', help="train dense layers")
parser.add_argument("--train2", action='store_true', help="train dense layers")
parser.add_argument("--predict", action='store_true', help="predict test data and save")
parser.add_argument("--sub", nargs=2, help="generate submission file")
parser.add_argument("--showconv", action='store_true', help="show summary of conv model")
parser.add_argument("--savelabel", action='store_true', help="show summary of conv model")

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
if args.genfirstfeats:
    print 'generating train and val features...'
    gen_vgg_features(gen_first=True)
    print 'done'
if args.train:
    print 'training dense layer...'
    train_bn_layers()
    print 'done'
if args.train2:
    print 'training dense layer...'
    train_bn_layers2()
    print 'done'
if args.savelabel:
    save_labels()
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
