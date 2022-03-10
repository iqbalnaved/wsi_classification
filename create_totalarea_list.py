import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import os, glob, shutil
import sys

dtype = sys.argv[1] #'train'

root_src = '/largeDataVolume/LeHou_GBMLGG_300x300/'

f = open('LeHou_GBMLGG_300x300_total_area_' + dtype + '.txt', 'a')

src = root_src + '/' + dtype + '/'   
dirlist = os.listdir(src)

i=0    
for dirname in dirlist: 

    fplist = glob.glob(os.path.join(src, dirname, '*.png'))
    
    for fp in fplist:
    
        img = cv2.imread(fp,0)
        binary_img = cv2.Canny(img,100,200) # edges

        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)

        if not props:
            total_area = 0
        else:
            total_area = sum([ele.area for ele in props])

        i = i + 1
        print('{}: {}, {}'.format(i+1, fp, total_area))                
        f.write('{}, {}, {}\n'.format(i+1, fp, total_area))

f.close()
