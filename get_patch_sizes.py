import os, glob
import sys

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000 


dtype = sys.argv[1] #'test'

dst = 'GBMLGG_patches_'+dtype+'.txt'
f = open(dst, 'w')
f.write('patchname, height, width\n')
patchdir = '/largeDataVolume/LeHou_GBMLGG_nonresized_patches/'+dtype # nonresized patches
dirlist = os.listdir(patchdir)
for dirname in dirlist:
    fpath_list = glob.glob(os.path.join(patchdir, dirname, '*.png'))
    for fpath in fpath_list:
        patchname = os.path.basename(fpath)
        patch = Image.open(fpath)
        w, h = patch.size
        print('{}, {}, {}'.format(patchname, h, w))
        f.write('{}, {}, {}\n'.format(patchname, h, w))
f.close() 
        
        

