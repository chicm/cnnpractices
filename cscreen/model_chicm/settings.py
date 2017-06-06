# Configure the INPUT_PATH to the directory which contains 'test', 'train' and 'additional'
# The addiontal training data should be put into INPUT_PATH/additional/Type_1,
# INPUT_PATH/additional/Type_2, and INPUT_PATH/additional/Type_3
# Please do NOT include slash('/') at the end of any following paths.
INPUT_PATH = '/home/chicm/ml/kgdata/cscreen'

# This directory is created automatically, normally you do not need to change this path
RESIZED_DATA_PATH = INPUT_PATH + '/resize640'

# This directory is create automatically, you do not need to change it.
STATGE1_TEST_DATA_PATH = RESIZED_DATA_PATH + '/test'

# This directory is create automatically, you do not need to change it.
MODEL_PATH = RESIZED_DATA_PATH + '/models'

# Set STATGE1_LABELED_TEST_DATA_PATH to the labled stage1 test data
#STATGE1_LABELED_TEST_DATA_PATH = 

# Before predicting stage2 test data, uncomment this and set it to the directory contains stage 2 test pictures.
#STATGE2_TEST_DATA_PATH = INPUT_PATH + '/test2'