import os
import numpy as np
import settings
MODEL_DIR = settings.MODEL_PATH

w_files_training = []
def save_weight(acc, modelname, epoch, num):
    f_name = '{}_{}_{:.5f}.pth'.format(modelname, epoch, acc)
    path = os.path.join(MODEL_DIR, f_name)
    if len(w_files_training) < num:
        w_files_training.append((acc, path))
        #save(model, path)
        return
    min = 10.0
    index_min = -1
    for i, item in enumerate(w_files_training):
        val_acc, fp = item
        if min > val_acc:
            index_min = i
            min = val_acc
    #print(min)
    if acc > min:
        #save(model, path)
        #os.remove(w_files_training[index_min][1])
        w_files_training[index_min] = (acc, path)
        


for epoch in range(4):
    acc = np.random.random()
    print(acc)
    save_weight(acc, 'res50', epoch, 3)

print(w_files_training)
print(w_files_training[0][1])