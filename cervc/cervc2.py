
import os, shutil, glob, sys
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from utils import *
from vgg16bn import *
from keras import applications
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

DATA_DIR = '/home/chicm/ml/cnnpractices/cervc/data/sample'
TRAIN_DIR = DATA_DIR+'/train'
TEST_DIR = DATA_DIR + '/test'
VALID_DIR = DATA_DIR + '/valid'
RESULT_DIR = DATA_DIR + '/results'

TRAIN_FEAT = RESULT_DIR + '/train_feat.dat'
VAL_FEAT = RESULT_DIR + '/val_feat.dat'
TEST_FEAT = RESULT_DIR + '/test_feat.dat'
WEIGHTS_FILE = RESULT_DIR + '/sf_weights.h5'
PREDICTS_FILE = RESULT_DIR + '/predicts'

batch_size = 32

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
    print(files[:10])

    for i in range(150):
        fn = files[i]
        #print TRAIN_DIR+'/'+fn
        shutil.move(TRAIN_DIR+'/'+fn, VALID_DIR+'/'+fn)

def create_model():
    vgg = Vgg16BN()
    vggmodel = vgg.model
    last_conv_idx = [i for i,l in enumerate(vggmodel.layers) if type(l) is Convolution2D] [-1]
    conv_layers = vggmodel.layers[:last_conv_idx+1]
    #print conv_layers
    bn_layers = [
        MaxPooling2D(input_shape = conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ]
    conv_layers.extend(bn_layers)
    #print conv_layers
    model = Sequential(conv_layers)
    model.compile(Adam(lr=0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])
    for i in range(last_conv_idx+1):
        model.layers[i].trainable = False
    print(model.summary())
    return model

def train():
    (val_classes, trn_classes, val_labels, trn_labels, val_filenames, trn_filenames, test_filenames) = get_classes(DATA_DIR+'/')
    gen_t = image.ImageDataGenerator(rotation_range=180, height_shift_range=0.05,
		shear_range=0.1, channel_shift_range=20, width_shift_range=0.1,
                horizontal_flip=True, vertical_flip=True)

    da_batches = get_batches(TRAIN_DIR, batch_size = batch_size, shuffle=False)
    val_batches = get_batches(VALID_DIR, batch_size = batch_size, shuffle=False)
    #est_batches = get_batches(TEST_DIR, batch_size = batch_size, shuffle=False)

    model = create_model()
    model.fit_generator(da_batches, samples_per_epoch=da_batches.samples*10, nb_epoch=2, 
                        validation_data=val_batches, nb_val_samples=val_batches.samples)
    model.optimizer.lr = 0.01
    model.fit_generator(da_batches, samples_per_epoch=da_batches.samples*10, nb_epoch=10, 
                        validation_data=val_batches, nb_val_samples=val_batches.samples)
    model.optimizer.lr = 0.00001
    model.fit_generator(da_batches, samples_per_epoch=da_batches.samples*10, nb_epoch=10, 
                        validation_data=val_batches, nb_val_samples=val_batches.samples)
    


def show_conv():
    model = create_model()
    print(model.summary())
    #model = get_vgg_model()
    #print model.summary()
    #bn_model = get_bn_model()
    #print bn_model.summary()
    



def save_predict():
    bn_model = get_bn_model()
    bn_model.load_weights(WEIGHTS_FILE)
    conv_test_feat = load_array(TEST_FEAT)
    preds = bn_model.predict(conv_test_feat, batch_size=batch_size)
    save_array(PREDICTS_FILE, preds)

def gen_submit(submit_filename, clip_percentage):
    preds = load_array(PREDICTS_FILE)
    print(preds[:20])
    subm = do_clip(preds, clip_percentage)
    subm_name = DATA_DIR+'/results/' + submit_filename

    trn_batches = get_batches(DATA_DIR+'/train', batch_size = batch_size)
    classes = sorted(trn_batches.class_indices, key=trn_batches.class_indices.get)
    #print classes
    batches = get_batches(DATA_DIR+'/test', batch_size = batch_size, shuffle=False)

    submission = pd.DataFrame(subm, columns=classes)
    submission.insert(0, 'image_name', [a[8:] for a in batches.filenames])
    #print [a for a in batches.filenames][:10]
    print(submission.head())
    submission.to_csv(subm_name, index=False, compression='gzip')


parser = argparse.ArgumentParser()
parser.add_argument("--mb", action='store_true', help="move validation data back to training directory")
parser.add_argument("--createval", action='store_true',
                    help="create validation data from training data")
parser.add_argument("--train", action='store_true', help="train dense layers")
parser.add_argument("--predict", action='store_true', help="predict test data and save")
parser.add_argument("--sub", nargs=2, help="generate submission file")
parser.add_argument("--showconv", action='store_true', help="show summary of conv model")

args = parser.parse_args()
if args.mb:
    print('moving back...')
    move_validation_back()
    print('done')
if args.createval:
    print('creating val data...')
    create_validation_data()
    print('done')
if args.train:
    print('training dense layer...')
    #train_bn_layers()
    train()
    print('done')
if args.predict:
    print('predicting test data...')
    save_predict()
    print('done')
if args.sub:
    print('generating submision file...')
    gen_submit(args.sub[0], (float)(args.sub[1]))
    print('done')
if args.showconv:
    show_conv()
