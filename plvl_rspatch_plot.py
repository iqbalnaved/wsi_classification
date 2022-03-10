# load image with 

# importing image object from PIL
# sample image
# TCGA-GBM/8c5308ed-4528-4dcc-b3eb-f3c382d490b0/TCGA-14-0789-01Z-00-DX7.0a15d486-2455-433b-bb96-155834c23b59.svs, (45088, 42386)
import math
import matplotlib.pyplot as plt
import os
import random 
import numpy as np

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

src = 'ps_region/vgg16_vgg16_rf/gbmlgg_300x300_100ppi_rs1_vgg16_vgg16_9_dist_per_image_rf_ps99.npz'
npzfile = np.load(src)
test_img_list = npzfile['img_list']
y_true = npzfile['y_test']
y_pred = npzfile['y_pred']

imlvl_pred_dict = {}
for i in range(len(test_img_list)):
    imgname = test_img_list[i]
    corr =  y_true[i] == y_pred[i]
    imlvl_pred_dict[imgname] = corr
    
src = 'LeHou_GBMLGG_300x300_100ppi_random_test_1.txt'
f = open(src)
content = f.readlines()
f.close()

patch_list = []
for line in content:
    fpath = line.strip().split(',')[0]
    patchname = os.path.basename(fpath)
    patch_list.append(patchname)

src = 'ps_output/vgg16/gbmlgg_300x300_100ppi_rs1_macroArch_vgg16_lr_rop_epoch_50_data_aug_patchlvl_rf_testset_predictions_after_patch_selection99.csv'
f = open(src)
content = f.readlines()
f.close()

corr_dict = {}
for line in content:
    fpath = line.strip().split(',')[0]
    patchname = os.path.basename(fpath)
    patch_list.append(patchname)
    y_true = int(line.strip().split(',')[1])
    y_pred = int(line.strip().split(',')[2])
    corr_dict[patchname] = y_true == y_pred
    
for i in range(10):
    rand_idx = random.randint(0,len(img_list)-1)
    imgname = img_list[rand_idx][0]
    img_w = img_list[rand_idx][1]
    img_h = img_list[rand_idx][2]
    patches_of_this_img = [patchname for patchname in patch_list if imgname in patchname]

    x_red, x_blue, x_green,y_red, y_blue, y_green = [],[], [],[], [],[]
    for patchname in patches_of_this_img:
        patchname = patchname.strip().split('.')[0] 
        top = int(patchname.split('_')[1])
        left = int(patchname.split('_')[2])
        if patchname in corr_dict:
            if corr_dict[patchname] == True: # patch_selected, correctly classified
                x_green.append(top)
                y_green.append(left)
            else:
                x_blue.append(top) # patch selected, misclassified
                y_blue.append(left)
        else:
            x_red.append(top) # patch regarded non-discriminative
            y_red.append(left)

    plt.scatter(x_green, y_green, c='green', alpha=0.5)
    plt.scatter(x_blue, y_blue, c='blue', alpha=0.5)
    plt.scatter(x_red, y_red, c='red', alpha=0.5)
    plt.scatter([0, 0, img_h, img_h], [0, img_w, 0, img_w], c='black')
    if imlvl_pred_dict[imgname] == True:
        plt.title('Image correctly classified using Vgg16+Vgg16+9-region RF model')
    else:
        plt.title('Image was mis-classified using Vgg16+Vgg16+9-region RF model')

    plt.savefig('plots/'+imgname+'_plvl_classification_after_patch_selection_plot.png', dpi=100)
    plt.show()
    
#    input('press any key..')
