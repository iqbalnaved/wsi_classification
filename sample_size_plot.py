import os, glob
import sys
import numpy as np

dtype = sys.argv[1] #'test'

src = 'GBMLGG_'+dtype+'.txt'
f = open(src)
content = f.readlines()
f.close()

img_list = []
for line in content:
    fpath = line.strip().split(',')[0]
    imgname = os.path.basename(fpath).split('.')[0]
#    img_w = int(line.strip().split(',')[1].strip(' ('))
#    img_h = int(line.strip().split(',')[2].strip(' )'))
#    img_list.append((imgname, img_w, img_h))
    img_list.append(imgname)
    
src = '/largeDataVolume/LeHou_GBMLGG_300x300/' + dtype

dirlist = os.listdir(src)

patch_list = []
for dirname in dirlist:
    fpath_list = glob.glob(os.path.join(src, dirname, '*.png'))
    for fpath in fpath_list:
        patch_list.append(os.path.basename(fpath)) 


def get_sample_size(N, Z=1.96, p = 0.5, e = 0.05):
#    e is the desired level of precision (i.e. the margin of error),
#    p is the (estimated) proportion of the population which has the attribute in question,
#    q is 1 – p. 
#    Z is the z-value is found in a Z-table.
#    n = sample size for known popluation N.
    n = ( (Z**2 * p * (1 - p)) / e**2 ) / ( 1 + ((Z**2 * p * (1 - p)) / (e ** 2 * N)) )
    return n

x,y = [],[]
i = 1
for imgname in img_list:
    patches_of_this_img = [patchname for patchname in patch_list if imgname in patchname]
    population_size = len(patches_of_this_img)
    sample_size = get_sample_size(population_size)
    x.append(population_size)
    y.append(sample_size)
    print('{}/{} {} , psize={}, ssize={}'.format(i, len(img_list), imgname, population_size, sample_size))
    i += 1

np.savez(dtype + 'set_sample_sizes.npz', img_list = img_list, psize=x, ssize=y)

############################

#import matplotlib.pyplot as plt
#import numpy as np
#npzfile = np.load(dtype+'set_sample_sizes.npz')
#img_list = npzfile['img_list']
#population_size_list = npzfile['psize']
#sample_size_list = npzfile['ssize']

#plt.scatter(population_size_list, sample_size_list) #, bins='auto')  # arguments are passed to np.histogram
#plt.xlabel('Population size')
#plt.ylabel('Sample size')
#plt.title("Plot of population size vs sample size for trainset.")
##Text(0.5, 1.0, "Histogram with 'auto' bins")
#plt.show()  


