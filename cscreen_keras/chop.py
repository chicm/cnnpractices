import utils; reload(utils)
from utils import *
#%matplotlib inline


DATA_DIR = '/home/chicm/data/cervc/chop400'
TRAIN_DIR = DATA_DIR+'/train'
TEST_DIR = DATA_DIR + '/test'
VALID_DIR = DATA_DIR + '/valid'
RESULT_DIR = DATA_DIR + '/results'
batch_size = 16
da = 5

gen_t = image.ImageDataGenerator(rotation_range=180,  horizontal_flip=True, zoom_range=0.1,
        shear_range=0.1, channel_shift_range=20, width_shift_range=0.1, height_shift_range=0.1, vertical_flip=True)

batches = get_batches(TRAIN_DIR, gen_t, batch_size=batch_size, target_size=(400,400))
val_batches = get_batches(VALID_DIR, batch_size=batch_size, shuffle = False, target_size=(400,400))
(val_classes, trn_classes, val_labels, trn_labels,
    val_filenames, filenames, test_filenames) = get_classes(DATA_DIR+'/')


conv_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def conv_preprocess(x):
    x = x - conv_mean
    return x[:, ::-1] # reverse axis rgb->bgr

def get_morden_conv_model():
    model = Sequential([
        Lambda(conv_preprocess, input_shape=(3, 400, 400)),
        Convolution2D(32, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32, 3, 3, activation='relu'),
        MaxPooling2D(),

        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        MaxPooling2D(),

        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        MaxPooling2D(),

        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        MaxPooling2D(),
        
        Flatten(),
        BatchNormalization(axis=1),
       
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

model = get_morden_conv_model()
print model.summary()

model.fit_generator(batches, samples_per_epoch=batches.nb_sample*da, nb_epoch=10, 
                        validation_data=val_batches, nb_val_samples=val_batches.nb_sample*da)

model.optimizer.lr = 0.01
model.fit_generator(batches, samples_per_epoch=batches.nb_sample*da, nb_epoch=10, 
                        validation_data=val_batches, nb_val_samples=val_batches.nb_sample*da)

model.optimizer.lr = 0.00001
model.fit_generator(batches, samples_per_epoch=batches.nb_sample*da, nb_epoch=30, 
                        validation_data=val_batches, nb_val_samples=val_batches.nb_sample*da)