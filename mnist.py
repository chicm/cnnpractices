#%%

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

import numpy as np

#%matplotlib inline
batch_size = 64


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape, y_train.shape

x_test = np.expand_dims(x_test, 1)
x_train = np.expand_dims(x_train, 1)

x_train.shape
#print y_train[:5]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print y_train[:5]
print y_test[:5]

#%%
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)

def norm_input(x):
    return x-mean_px/std_px


#%%
gen = image.ImageDataGenerator()
batches = gen.flow(x_train, y_train, batch_size=64)
test_batches = gen.flow(x_test, y_test, batch_size=64)

#%%
def get_lin_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1, 28,28)),
        Flatten(),
        Dense(10, activation = 'softmax')
    ])
    model.compile(Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def fit_lin_model():
    lm = get_lin_model()

    lm.fit(x_train, y_train, validation_data = (x_test, y_test),  nb_epoch = 4)
    lm.optimizer.lr = 0.1
    lm.fit(x_train, y_train, validation_data = (x_test, y_test),  nb_epoch = 4)
    lm.optimizer.lr = 0.01
    lm.fit(x_train, y_train, validation_data = (x_test, y_test), nb_epoch = 4)


#%%
def get_fc_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1, 28,28)),
        Flatten(),
        Dense(512, activation = 'softmax'),
        Dense(10, activation = 'softmax')
    ])
    model.compile(Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def fit_fc_model():
    fm = get_fc_model()

    fm.fit_generator(batches, batches.n, nb_epoch=1, 
                        validation_data=test_batches, nb_val_samples=test_batches.n)

    fm.optimizer.lr = 0.1
    fm.fit_generator(batches, batches.n, nb_epoch=4, 
                        validation_data=test_batches, nb_val_samples=test_batches.n)

    fm.optimizer.lr = 0.01
    fm.fit_generator(batches, batches.n, nb_epoch=4, 
                        validation_data=test_batches, nb_val_samples=test_batches.n)


def get_conv_model():
    model = Sequential([
        Lambda(norm_input, input_shape = (1, 28, 28)),
        Convolution2D(32, 3, 3, activation='relu'),
        Convolution2D(32, 3, 3, activation='relu'),
        MaxPooling2D(),
        Convolution2D(64, 3, 3, activation='relu'),
        Convolution2D(64, 3, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(Adam(), loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model

def train_conv_model():
    model = get_conv_model()
    model.fit_generator(batches, batches.n, nb_epoch = 1, validation_data=test_batches, nb_val_samples=test_batches.n)
    model.optimizer.lr = 0.1

    model.fit_generator(batches, batches.n, nb_epoch = 1, validation_data=test_batches, nb_val_samples=test_batches.n)

    model.optimizer.lr = 0.01
    model.fit_generator(batches, batches.n, nb_epoch = 8, validation_data=test_batches, nb_val_samples=test_batches.n)


def train_da_conv_model():
    gen = image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, 
            height_shift_range=0.08, zoom_range=0.08)
    batches = gen.flow(x_train, y_train, batch_size=64)
    #test_batches = gen.flow(x_test, y_test, batch_size=64)
    model = get_conv_model()
    model.fit_generator(batches, batches.n, nb_epoch = 1, validation_data=test_batches, nb_val_samples=test_batches.n)
    model.optimizer.lr = 0.1

    model.fit_generator(batches, batches.n, nb_epoch = 1, validation_data=test_batches, nb_val_samples=test_batches.n)

    model.optimizer.lr = 0.01
    model.fit_generator(batches, batches.n, nb_epoch = 8, validation_data=test_batches, nb_val_samples=test_batches.n)

    model.optimizer.lr = 0.001
    model.fit_generator(batches, batches.n, nb_epoch = 10, validation_data=test_batches, nb_val_samples=test_batches.n)

    model.optimizer.lr = 0.0001
    model.fit_generator(batches, batches.n, nb_epoch = 10, validation_data=test_batches, nb_val_samples=test_batches.n)

#train_da_conv_model()

def get_morden_conv_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1, 28, 28)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32, 3, 3, activation='relu'),
        MaxPooling2D(),

        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        MaxPooling2D(),

        BatchNormalization(axis=1),
        Convolution2D(128, 3, 3, activation='relu'),
        MaxPooling2D(),
        
        Flatten(),
        BatchNormalization(axis=1),
       
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    model.compile(Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def train_morden_conv_model():
    gen = image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3, 
            height_shift_range=0.08, zoom_range=0.08)
    batches = gen.flow(x_train, y_train, batch_size=64)
    #test_batches = gen.flow(x_test, y_test, batch_size=64)
    model = get_morden_conv_model()
    model.fit_generator(batches, batches.n, nb_epoch = 1, validation_data=test_batches, nb_val_samples=test_batches.n)
    model.optimizer.lr = 0.1

    model.fit_generator(batches, batches.n, nb_epoch = 4, validation_data=test_batches, nb_val_samples=test_batches.n)

    model.optimizer.lr = 0.01
    model.fit_generator(batches, batches.n, nb_epoch = 8, validation_data=test_batches, nb_val_samples=test_batches.n)

    model.optimizer.lr = 0.0001
    model.fit_generator(batches, batches.n, nb_epoch = 8, validation_data=test_batches, nb_val_samples=test_batches.n)

train_morden_conv_model()
