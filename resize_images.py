import os,glob
import sys

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
dtype = sys.argv[1] 

print('processing {}'.format(dtype))
src = '/largeDataVolume/LeHou_GBMLGG/' + dtype
dirlist = os.listdir(src)
c = 0
dst = '/largeDataVolume/LeHou_GBMLGG_300x300/' + dtype
for dname in dirlist:
    fpath_list = glob.glob(os.path.join(src, dname, '*.png'))
    for fpath in fpath_list:
        fname = os.path.basename(fpath)
        if not os.path.isfile(os.path.join(dst, dname, fname)):
            im = Image.open(fpath)
            im = im.resize((300,300))
            if not os.path.isdir(os.path.join(dst, dname)):
                os.makedirs(os.path.join(dst, dname))
            im.save(os.path.join(dst, dname, fname))
            print('{}: {} resized to {}x{} and saved to {}/{}'.format(c,fname, im.width, im.height, dst,dname))
        else:
            print('{}: {} already resized'.format(c, fname))
        c = c + 1
print('total resized: {}'.format(c))


