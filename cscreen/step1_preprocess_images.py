import cfg
import os, shutil, glob, sys, cv2
import numpy as np

ADD_DIR = cfg.INPUT_DIR + '/additional'
NEW_DATA_DIR = cfg.INPUT_DIR + '/resize640_2'

blacklist = ['Type_2/2845.jpg', 'Type_2/5892.jpg', 'Type_1/5893.jpg',
    'Type_1/1339.jpg', 'Type_1/3068.jpg', 'Type_2/7.jpg',
    'Type_1/746.jpg', 'Type_1/2030.jpg', 'Type_1/4065.jpg',
    'Type_1/4702.jpg', 'Type_1/4706.jpg', 'Type_2/1813.jpg', 'Type_2/3086.jpg']
files_0522 = ['/Type_2/80.jpg', '/Type_3/968.jpg', '/Type_3/1120.jpg']

def create_directories():
    try_mkdir(NEW_DATA_DIR)
    try_mkdir(NEW_DATA_DIR+'/train')
    try_mkdir(NEW_DATA_DIR+'/train/Type_1')
    try_mkdir(NEW_DATA_DIR+'/train/Type_2')
    try_mkdir(NEW_DATA_DIR+'/train/Type_3')
    try_mkdir(NEW_DATA_DIR+'/valid')
    try_mkdir(NEW_DATA_DIR+'/valid/Type_1')
    try_mkdir(NEW_DATA_DIR+'/valid/Type_2')
    try_mkdir(NEW_DATA_DIR+'/valid/Type_3')
    try_mkdir(NEW_DATA_DIR+'/test')
    try_mkdir(NEW_DATA_DIR+'/test/unknown')
    try_mkdir(NEW_DATA_DIR+'/results')
    try_mkdir(NEW_DATA_DIR+'/models')

def try_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

def resize_640(src_dir, tgt_dir, match_str):
    os.chdir(src_dir)
    files = glob.glob(match_str)
    for f in files:
        if f in blacklist:
            print('skipping {}'.format(f))
            continue
        fn = src_dir+'/'+f
        print(fn)
        tgt_fn = tgt_dir+'/'+f

        if os.path.exists(tgt_fn):
            split = f.split('.')
            tgt_fn = tgt_dir + '/' + split[0] + '_add.' + split[1]
            print(tgt_fn)

        img = cv2.imread(fn)
        res = cv2.resize(img, (640, 640))
        cv2.imwrite(tgt_fn, res)


def create_additional_copies(num_copies):   
    src_dir = cfg.INPUT_DIR + '/train'
    tgt_dir = NEW_DATA_DIR + '/train'
    val_dir = NEW_DATA_DIR + '/valid'
    os.chdir(val_dir)
    val_files = glob.glob('*/*.jpg')
    os.chdir(src_dir)
    src_files = glob.glob('*/*.jpg')
    count = 0
    for fn in src_files:
        if fn in blacklist:
            continue
        if fn in files_0522:
            continue
        if fn in val_files:
            print('skip, already in validation files: {}'.format(fn))
            continue
        
        img = cv2.imread(src_dir + '/' + fn)
        res = cv2.resize(img, (640, 640))

        split = fn.split('.')
        for i in range(num_copies):
            fn_copy = split[0] + '_copy_' + str(i) + '.' + split[1]
            tgt_copy = tgt_dir + '/' + fn_copy
            print(tgt_copy)
            if not os.path.exists(tgt_copy):
                print('creating copy: {}'.format(tgt_copy))
                cv2.imwrite(tgt_copy, res)
                count += 1
    print('created {} additional copies'.format(count))

def resize_images():
    resize_640(cfg.INPUT_DIR+'/train', NEW_DATA_DIR+'/train', '*/*.jpg')
    resize_640(cfg.INPUT_DIR+'/additional', NEW_DATA_DIR+'/train', '*/*.jpg')
    resize_640(cfg.INPUT_DIR+'/test', NEW_DATA_DIR+'/test/unknown', '*.jpg')

def find_dup_names():
    os.chdir(ADD_DIR)
    add_files = glob.glob('*/*')
    os.chdir(cfg.INPUT_DIR+'/train')
    files = glob.glob('*/*')
    count = 0
    for f in add_files:
        if f in files:
            count += 1
    print(count)

def update_label_0522():
    train_dir = NEW_DATA_DIR+'/train'
    shutil.move(train_dir+'/Type_2/80.jpg', train_dir+'/Type_3/80.jpg')
    shutil.move(train_dir+'/Type_3/968.jpg', train_dir+'/Type_1/968_add.jpg')
    shutil.move(train_dir+'/Type_3/1120.jpg', train_dir+'/Type_1/1120.jpg')

def create_validation_data():
    train_dir = NEW_DATA_DIR+'/train'
    val_dir = NEW_DATA_DIR+'/valid'

    os.chdir(train_dir)
    files = glob.glob('*/*.jpg')
    files = np.random.permutation(files)

    for i in range(600):
        fn = files[i]
        shutil.move(train_dir+'/'+fn, val_dir+'/'+fn)

def move_back_validation_data():
    train_dir = NEW_DATA_DIR+'/train'
    val_dir = NEW_DATA_DIR+'/valid'

    os.chdir(val_dir)
    files = glob.glob('*/*.jpg')

    for fn in files:
        shutil.move(val_dir+'/'+fn, train_dir+'/'+fn)


if __name__ == "__main__":
    if False:
        print('creating directories')
        create_directories()
        print('done')
    if False:
        print('creating resized images, this will take a while...')
        resize_images()
        print('done')
    if False:
        update_label_0522()
        print('done')
    if False:
        print('creating validation data')
        try:
            move_back_validation_data()
        except:
            pass
        create_validation_data()
        print('done')
    if True:
        create_additional_copies(4)

