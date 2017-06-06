# Configure the TRAIN_DATA_PATH to the directory which contains 'test', 'train' and 'additional'
# The addiontal training data should be put into TRAIN_DATA_PATH/additional/Type_1,
# TRAIN_DATA_PATH/additional/Type_2, and TRAIN_DATA_PATH/additional/Type_3
# Please do NOT include slash('/') at the end of any following paths.
TRAIN_DATA_PATH = '/home/chicm/ml/kgdata/cscreen'

# Normally you do not need to change this value
RESIZED_DATA_PATH = TRAIN_DATA_PATH + '/val300-r2'

# TEST_DATA_PATH is preconfigured to predict the stage 1 test data. If you want to predict stage 1
# test data, you do not need to change it.
# Config this TEST_DATA_PATH when you want to predict new test data, make sure there is an "unknown"
# directory under this TEST_DATA_PATH and jpg files should be put into TEST_DATA_PATH/unknown
TEST_DATA_PATH = RESIZED_DATA_PATH + '/test'

# Normally you do not need to change this value
MODEL_PATH = RESIZED_DATA_PATH + '/models'
