# open svs file, given in LeHou's list'
# extract level 0 image in tiles of size 300x300
# save them in jpg format 

import openslide
import os, glob
import sys
import cv2
import numpy as np
#dtypes = ['train', 'test']

svs_src = '..' #'/media/adjeroh/10TBHD/DATABASES/Cancer_Datasets'

level = 0  # pyramid level

#for dtype in dtypes:

dtype = sys.argv[1]
tile_sz = sys.argv[2] # dim square tile size
start_idx = int(sys.argv[3])
end_idx = int(sys.argv[4])

src = 'LeHou_GBMLGG_'+dtype+'.txt'
f = open(src)
content = f.readlines()
f.close()

if end_idx == -1:
    end_idx = len(content)

print('start={}, end={}, len(content)={}'.format(start_idx, end_idx, len(content)))

if tile_sz == '300':
    size = (300, 300) # size of the tile
    vstride, hstride = (300, 300)
    savedir = 'LeHou_GBMLGG_300x300'
    resize=False

if tile_sz == '600':
    size = (600, 600) # size of the tile
    vstride, hstride = (600, 600)
    savedir = 'LeHou_GBMLGG_600x600'
    resize = True
    rsz_dim = (300,300)

if tile_sz == '1200':
    size = (1200, 1200) # size of the tile
    vstride, hstride = (1200, 1200)
    savedir = 'LeHou_GBMLGG_1200x1200'
    resize = True
    rsz_dim = (300,300)

print(savedir)

#lc = 0
for i in range(start_idx, end_idx):
#    found = False
    line = content[i]
#    lc = lc + 1
    fpath = line.strip().split(',')[0]
    filename = os.path.basename(fpath)
    label = line.strip().split(',')[1].strip()
    wsi = openslide.OpenSlide(os.path.join(svs_src, fpath))
    print('{}/{}: {} size={}'.format(i, end_idx, filename, wsi.dimensions))

    width, height =  wsi.dimensions # A (width, height) tuple for level 0 of the slide.

    #for key in wsi.properties:
    #    print('{}: {}'.format(key, wsi.properties[key]))
    dst = '../' + savedir + '/' + dtype + '/' + label + '/'
    if not os.path.isdir(dst):
        os.makedirs(dst)

    c = 0
    for top in range(0, height - size[0], vstride):
        for left in range(0, width - size[1], hstride):
            location = (top,left) # top left pixel
            save_path = os.path.join(dst, filename.split('.')[0] + '_' + str(top) + '_' + str(left) + '.png')
            c = c + 1            
#            if not os.path.isfile(save_path):
            tile = wsi.read_region(location, level, size)
            if resize == True:
                img = cv2.resize(np.array(tile), rsz_dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(save_path, img)
            else:
                tile.save(save_path)
#                print('\tsave#{}: {}'.format(c, filename.split('.')[0] + '_' + str(top) + '_' + str(left) + '.png'))
#            else:
#                print('\tskipping#{}: {}'.format(c, filename.split('.')[0] + '_' + str(top) + '_' + str(left) + '.png'))
#            print('\t\tsave#{}: {}'.format(c, filename.split('.')[0] + '_' + str(top) + '_' + str(left) + '.png'))
            
    print('{} patches saved for {}'.format(c, filename))        

                
