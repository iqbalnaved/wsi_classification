import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import os, glob
from shutil import copyfile
import sys

dtype = sys.argv[1]

dst = '/largeDataVolume/LeHou_GBMLGG_300x300_'+dtype+'set_blanks/'


f = open('LeHou_GBMLGG_300x300_blanks_'+dtype+'.txt')
content = f.readlines()
f.close()

i=0
for line in content:
    
    fp = line.strip()

    parentdir = os.path.dirname(fp).split('/')[-1]
    filename = os.path.basename(fp)
    
    if not os.path.isdir(os.path.join(dst, parentdir)):
        os.makedirs(os.path.join(dst, parentdir))

#    copyfile(fp, os.path.join(dst, parentdir, filename))        
    os.rename(fp, os.path.join(dst, parentdir, filename))
    
    i = i + 1
    
    print('{}/{} copying  {} as {}'.format(i, len(content), filename, parentdir + '/' + filename))                


