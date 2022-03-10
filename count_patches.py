import os, glob
import sys

dtype = sys.argv[1] 

dataset_sizes = {'train': 842, 'test': 217}

src = '/largeDataVolume/LeHou_GBMLGG/' + dtype

dirlist = os.listdir(src)

already_checked = []
c = 0
for dirname in dirlist:
    files_list = glob.glob(os.path.join(src, dirname, '*.png'))
    for fpath in files_list:
        imgname = os.path.basename(fpath).split('_')[0]
        if imgname not in already_checked:
            already_checked.append(imgname)
            c = c + 1
            patches_list = glob.glob(os.path.join(src, dirname, imgname + '*.png'))
            print('{}-set {}/{}: {} has {} patches'.format(dtype, c, dataset_sizes[dtype], imgname, len(patches_list)))
            
print('total number of {}-set images patch extracted: {}'.format(dtype, len(already_checked)))
            
