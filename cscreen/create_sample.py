import settings
import os, shutil, glob, sys, cv2
import numpy as np

SAMPLE_DIR = settings.RESIZED_DATA_PATH + '/samples'

blacklist = ['Type_2/2845.jpg', 'Type_2/5892.jpg', 'Type_1/5893.jpg',
    'Type_1/1339.jpg', 'Type_1/3068.jpg', 'Type_2/7.jpg',
    'Type_1/746.jpg', 'Type_1/2030.jpg', 'Type_1/4065.jpg',
    'Type_1/4702.jpg', 'Type_1/4706.jpg', 'Type_2/1813.jpg', 'Type_2/3086.jpg']
files_0522 = ['/Type_2/80.jpg', '/Type_3/968.jpg', '/Type_3/1120.jpg']


def create_sample_directories():
    try_mkdir(SAMPLE_DIR)
    try_mkdir(SAMPLE_DIR+'/train')
    try_mkdir(SAMPLE_DIR+'/train/Type_1')
    try_mkdir(SAMPLE_DIR+'/train/Type_2')
    try_mkdir(SAMPLE_DIR+'/train/Type_3')
    try_mkdir(SAMPLE_DIR+'/valid')
    try_mkdir(SAMPLE_DIR+'/valid/Type_1')
    try_mkdir(SAMPLE_DIR+'/valid/Type_2')
    try_mkdir(SAMPLE_DIR+'/valid/Type_3')
    try_mkdir(SAMPLE_DIR+'/results')
    try_mkdir(SAMPLE_DIR+'/models')

def try_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

def resize_640(src_dir, tgt_dir, match_str, limits=0):
    os.chdir(src_dir)
    files = glob.glob(match_str)

    if limits > 0:
        files = np.random.permutation(files)
        files = files[:limits]

    for f in files:
        if f in blacklist:
            continue
        fn = src_dir+'/'+f
        print('.', end='',flush=True)
        tgt_fn = tgt_dir+'/'+f

        if os.path.exists(tgt_fn):
            split = f.split('.')
            tgt_fn = tgt_dir + '/' + split[0] + '_add.' + split[1]
            #print(tgt_fn)

        img = cv2.imread(fn)
        res = cv2.resize(img, (640, 640))
        cv2.imwrite(tgt_fn, res)


def resize_images():
    #resize_640(settings.TRAIN_DATA_PATH+'/additional', SAMPLE_DIR+'/valid', '*/*.jpg', 500)
    resize_640(settings.TRAIN_DATA_PATH+'/train', SAMPLE_DIR+'/train', '*/*.jpg')
    

def create_validation_data():
    train_dir = SAMPLE_DIR+'/train'
    val_dir = SAMPLE_DIR+'/valid'

    os.chdir(train_dir)
    files = glob.glob('*/*.jpg')
    files = np.random.permutation(files)

    for i in range(150):
        fn = files[i]
        shutil.move(train_dir+'/'+fn, val_dir+'/'+fn)

if __name__ == "__main__":
    if True:
        print('creating directories')
        create_sample_directories()
        print('done')
    if True:
        print('creating resized images, this will take a while...')
        resize_images()
        print('done')
    if True:
        print('creating validation data')
        create_validation_data()
        print('done')

