# open svs file, given in LeHou's list'
# extract level 0 image in tiles of size 300x300
# save them in jpg format 

import openslide
import os, glob
import sys

dtype = sys.argv[1]

svs_src = '/largeDataVolume/'

level = 0  # pyramid level

src = '/largeDataVolume/LeHou_GBMLGG_'+dtype+'.txt'
f = open(src)
content = f.readlines()
f.close()

#lc = 0 
#for i in range(len(content)-1, -1, -1):
for i in range(819, -1, -1):
#    found = False
    line = content[i]
#    lc = lc + 1
    fpath = line.strip().split(',')[0]
    filename = os.path.basename(fpath)
    label = line.strip().split(',')[1].strip()
    wsi = openslide.OpenSlide(os.path.join(svs_src, fpath))
    print('{}-th: {} size={}'.format(i+1, filename, wsi.dimensions))

    width, height =  wsi.dimensions # A (width, height) tuple for level 0 of the slide.

    #for key in wsi.properties:
    #    print('{}: {}'.format(key, wsi.properties[key]))
    dst = '/largeDataVolume/LeHou_GBMLGG/' + dtype + '/' + label + '/'
    if not os.path.isdir(dst):
        os.makedirs(dst)

    vstride, hstride = (int(height/10.), int(width/10.))

    size = (int(height/10.), int(width/10.)) # size of the tile
    c = 0
    for top in range(0, height - size[0], vstride):
        for left in range(0, width - size[1], hstride):
            location = (top,left) # top left pixel
            save_path = os.path.join(dst, filename.split('.')[0] + '_' + str(top) + '_' + str(left) + '.png')
            c = c + 1            
            if not os.path.isfile(save_path):
                tile = wsi.read_region(location, level, size)
                tile.save(save_path)
                print('\t\tsave#{}: {}'.format(c, filename.split('.')[0] + '_' + str(top) + '_' + str(left) + '.png'))
            else:
                print('\t\tskipping#{}: {}'.format(c, filename.split('.')[0] + '_' + str(top) + '_' + str(left) + '.png'))
#                found = True
#                break
#        if found:
#            break
#    if found:
#        break

            
    print('{}-th: {} patches saved for {}'.format(i+1, c, filename))        

                
