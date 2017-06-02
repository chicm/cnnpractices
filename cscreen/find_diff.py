import os, glob

def find_diff_names(DIR1, DIR2):
    os.chdir(DIR1+'/train')
    files1 = glob.glob('*/*.jpg')
    print(len(files1))    
    os.chdir(DIR1+'/valid')
    files1.extend(glob.glob('*/*.jpg'))
    files1 = set(files1)
    print(len(files1))
    os.chdir(DIR2+'/train')
    files2 = glob.glob('*/*.jpg')
    os.chdir(DIR2+'/valid')
    files2.extend(glob.glob('*/*.jpg'))
    files2 = set(files2)
    not_2 = []
    not_1 = []
    for f in files1:
        if not f in files2:
            not_2.append(f)
    
    for f in files2:
        if not f in files1:
            not_1.append(f)

    return not_2, not_1

d1 = '/home/chicm/ml/kgdata/cscreen/resize640'
d2 = '/home/chicm/ml/kgdata/cscreen/resize640_3'

l1, l2 = find_diff_names(d1, d2)

print('only in {}'.format(d1))
print(l1)

print('only in {}'.format(d2))
print(l2) 

