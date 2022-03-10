# open svs file, given in LeHou's list'
# extract level 0 image in tiles of size 300x300
# save them in jpg format 

import openslide
import os, glob

src = '/media/adjeroh/10TBHD/DATABASES/Cancer_Datasets/LeHou/GBMLGG_test.txt'
f = open(src)
content = f.readlines()
f.close()

svs_src = '/media/adjeroh/10TBHD/DATABASES/Cancer_Datasets/TCGA-GBM'
dirlist = os.listdir(svs_src)
filepath_list = []
for dirname in dirlist:
    fpath = glob.glob(os.path.join(svs_src, dirname, '*.svs'))
    filepath_list.append(fpath[0])

for line in content:
    filename = line.strip().split(' ')[0]
    label = line.strip().split(' ')[1]
    for fpath in filepath_list:
        if filename in fpath:
           wsi = openslide.OpenSlide(fpath)
           print('{}, {}, {}'.format(filename, fpath, wsi.dimensions))
    
# (36001, 25891)
#print('{}\n{}\n{}'.format(wsi.detect_format(src), wsi.level_count, wsi.dimensions))

#for key in wsi.properties:
#    print('{}: {}'.format(key, wsi.properties[key]))


#location = (0,0) # top left pixel
#level = 0
#size = (300, 300) # size of the tile
#tile = wsi.read_region(location, level, size)

#tile.save('tile.png')
