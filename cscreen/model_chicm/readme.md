## OS and Hardware requirement
This model was developed on following software and hardware platform:
+ GTX 1080ti GPU with 11GB GPU Memory
+ Ubuntu 16.04
+ Anaconda 3 and Python 3.6 with following packages:
  - cv2
  - pytorch
  - torchvision
  - bcolz

This model has not been tested on other GPU or OS.

## Steps to train and create submission file:
1. Prepare the data, download all the 7z files put them into a single directory, let's call this directory as TRAIN_DATA_PATH. Create an subdirectory "additional" under TRAIN_DATA_PATH and move the additional_Type_1_v2.7z, additional_Type_2_v2.7z, additional_Type_3_v2.7z files into TRAIN_DATA_PATH/additional directory. Then unpack all the 7z files.
Now under your TRAIN_DATA_PATH you should have "train", "test" and "additional" subdirectories. 
```
chicm@chicm-ubuntu:~/ml/kgdata/input$ ls
additional  test  train
```
Under "additional" directory you should have "Type_1", "Type_2" and "Type_3" subdirectories.
```
chicm@chicm-ubuntu:~/ml/kgdata/input/additional$ ls
Type_1  Type_2  Type_3
```
2. Open settings.py, change the value of TRAIN_DATA_PATH to the absolute path of your TRAIN_DATA_PATH, without a slash '/' at the end.

3. Enter the directory where the step1_preprocess_images.sh is located, Run ./step1_preprocess_images.sh, This step normally take around half an hour on my computer.
```
./step1_preprocess_images.sh
```
Please note, the above script must be executed from the directory where it is located. 

4. Run ./step2_train_models.sh, This step normally take around 20 hours on my GTX1080ti GPU.
```
./step2_train_models.sh
```
5. Run ./step3_create_submission.sh, Then you will get the submission file submit1.csv in TRAIN_DATA_PATH/resize640/results directory.
```
./step3_create_submission.sh
```
## Steps to make predition on new test data:
0. To predict new test data, you must firstly finish the 1-4 steps above.
1. Prepare test data: Reference the directory structure of TRAIN_DATA_PATH/resize640/test,  there is a "unknown" subdirectory under test directory, and all the jpg files are in the "unknown" directory. Prepare your new test data the same way as TRAIN_DATA_PATH/resize640/test. Create a directory, let's call this directory as TEST_DIR, then create "unknown" subdirectory under TEST_DIR, then put test jpg files into TEST_DIR/unknown/.
Open settings.py,  set the value of TEST_DATA_PATH to the absolute path of your TEST_DIR, do not include the "unknown" at the end. 
2. Run ./step3_create_submission.sh, Then you will still get the submission file submit1.csv in TRAIN_DATA_PATH/resize640/results directory.
```
./step3_create_submission.sh
```
