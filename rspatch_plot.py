# load image with 

# importing image object from PIL
# sample image
# TCGA-GBM/8c5308ed-4528-4dcc-b3eb-f3c382d490b0/TCGA-14-0789-01Z-00-DX7.0a15d486-2455-433b-bb96-155834c23b59.svs, (45088, 42386)
import math
import matplotlib.pyplot as plt
import os
import random 

src = 'GBMLGG_test.txt'
f = open(src)
content = f.readlines()
f.close()

img_list = []
for line in content:
    fpath = line.strip().split(',')[0]
    imgname = os.path.basename(fpath).split('.')[0]
    img_w = int(line.strip().split(',')[1].strip(' ('))
    img_h = int(line.strip().split(',')[2].strip(' )'))
    img_list.append((imgname, img_w, img_h))

#src = 'LeHou_GBMLGG_300x300_100ppi_random_test_1.txt'
src = 'LeHou_GBMLGG_300x300_adaptive_ppi_gridwise_random_test_1.txt'
f = open(src)
content = f.readlines()
f.close()

patch_list = []
for line in content:
    fpath = line.strip().split(',')[0]
    patchname = os.path.basename(fpath)
    patch_list.append(patchname)

# test set
sample_image_list = ['TCGA-S9-A6WE-01Z-00-DX1', 'TCGA-S9-A6WI-01Z-00-DX1', 'TCGA-S9-A6WL-01Z-00-DX1', 'TCGA-S9-A6WO-01Z-00-DX1', 
'TCGA-S9-A7IQ-01Z-00-DX1', 'TCGA-S9-A7J1-01Z-00-DX1', 'TCGA-S9-A7QY-01Z-00-DX1', 'TCGA-TM-A7CF-01Z-00-DX1', 'TCGA-TM-A7CF-01Z-00-DX2']

    
#for i in range(10):
#    rand_idx = random.randint(0,len(img_list)-1)
#    imgname = img_list[rand_idx][0]
#    img_w = img_list[rand_idx][1]
#    img_h = img_list[rand_idx][2]
for imgname in sample_image_list:
    patches_of_this_img = [patchname for patchname in patch_list if imgname in patchname]

    x,y = [],[]
    for patchname in patches_of_this_img:
        patchname = patchname.strip().split('.')[0] 
        top = int(patchname.split('_')[1])
        left = int(patchname.split('_')[2])
        x.append(top)
        y.append(left)

    plt.scatter(x,y, c='blue', alpha=0.5)
    plt.scatter([0, 0, img_h, img_h], [0, img_w, 0, img_w], color="red")
    plt.title(imgname)
    plt.savefig('plots/'+imgname+'_gridwise_sample_patch_plot.png', dpi=100)
    plt.show()
    
#    input('press any key..')
