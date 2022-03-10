import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import os, glob, shutil
import sys

dtype = sys.argv[1] #'train'

root_src = '/largeDataVolume/LeHou_GBMLGG_300x300/'
root_dst = '/largeDataVolume/LeHou_GBMLGG_300x300_blanks/'

threshold = 1339.15

f = open('LeHou_GBMLGG_300x300_blanks_' + dtype + '.txt', 'w')

src = root_src + '/' + dtype + '/'   
dirlist = os.listdir(src)

i=0    
for dirname in dirlist: 

    fplist = glob.glob(os.path.join(src, dirname, '*.png'))
    
    for fp in fplist:

        parentdir = os.path.dirname(fp).split('/')[-1]
        filename = os.path.basename(fp)

        img = cv2.imread(fp,0)
        binary_img = cv2.Canny(img,100,200) # edges

        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)

        if not props:
    #        print('{}/{}: empty props: mostly blank'.format(i+1, len(fplist)))
            total_area = 0
        else:
            total_area = sum([ele.area for ele in props])

        if total_area < threshold:
#            if not os.path.isdir(os.path.join(dst)):
#                os.makedirs(os.path.join(dst))
            i = i + 1
#            os.rename(os.path.join(fp), os.path.join(dst, parentdir, filename))
            print('{}: {}/{} total_area={}'.format(i+1, parentdir, filename, total_area))                
            f.write('{}\n'.format(fp))

f.close()
