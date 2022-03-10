# ChangeLog:
# We cannot add probabilities, but can add the max(P) to get the 
# overall frequency within a region or within the whole image, and normalize. 
# Create features for SVM training WITHOUT patch selection.

import pickle
import os, glob
import numpy as np
import sys
# create image level distributions for training set.

#def get_norm_neighbor_class_histogram(neighobor_dict, pred_dict, filename, num_classes=8):
## gets a normalized histogram of class predictions by SVM from the neighobring patches
#    class_dist = [0.0] * num_classes
#    for neighborpatchname in neighbor_dict[filename]:
#        pred = pred_dict[neighborpatchname][2]
#        class_dist[pred] +=1.
#    sum_class_dist = sum(class_dist)    
#    norm_class_dist = [ float(x) / sum_class_dist for x in class_dist]

#    return norm_class_dist
#    
#print('loading neighborhood dictionary')

## contains both train and valset neighbor info
#src = 'output/rail_images_300x300_wloc_neighbor_list.pkl'
#        
#with open(src, 'rb') as f:
#   neighbor_dict = pickle.load(f)        

print('loading data to dictionary...')
# pred_dict format 
# pred_dict[patchname] = {[softmax_output], ground_truth, pred_label, entropy, max_prob-sec_max_prob, neighbor_mean_jsd, neighbor_std_jsd}
# change: softmax_outout is now calculated by getting the normalized class histogram
# obtained from the patch neighborhood
dbname = 'gbmlgg_300x300_adaptive_ppi_gridrs1'
cnn = sys.argv[1] #'vgg16' #'resnet50' #
# patchname, ground_truth, predicted label, softmax
src = 'features/'+dbname+'_macroArch_'+cnn+'_data_aug_lr_rop_epoch50_net_valset_smx_predictions.txt'

f = open(src)
content = f.readlines()
f.close()

pred_dict = {}
for line in content:
    parts = line.strip().split(',')
    filename = parts[0]
    gt = int(parts[1])
    pred = int(parts[2])    
    softmax =  [float(x) for x in parts[3:]]
    pred_dict[filename] = [softmax, gt, pred] # [] keesp a space for normalized histogram

#for filename in pred_dict:
#    norm_class_dist = get_norm_neighbor_class_histogram(neighbor_dict, pred_dict, filename)
#    pred_dict[filename][0] = norm_class_dist


src = '/largeDataVolume/cancer_project/GBMLGG_test.txt' #'/media/adjeroh/10TBHD/DATABASES/Rail_Surface_Database/rail_images_trainval/train'
f = open(src)
content = f.readlines()
f.close()
unfiltered_img_list =  [] # image list before applying blank removal, all patches of an image may be removed during cleaning (one case found) 
unfiltered_img_hw_list = []
for line in content:    
    fpath = line.split(',')[0].strip()
    imgname = os.path.basename(fpath).split('.')[0]
    sz = line.split(',')[1:]
    sz = [int(x.strip().strip('()')) for x in sz]
    h = int(sz[0])
    w = int(sz[1])
    unfiltered_img_list.append(imgname)
    unfiltered_img_hw_list.append((h,w)) 
    
src = 'LeHou_GBMLGG_300x300_adaptive_ppi_gridwise_random_test_1.txt'
f = open(src)
content = f.readlines()[1:]
f.close()
img_list, img_sz_list, patch_list = [], [], []
for line in content:
    patchname = os.path.basename(line.split(',')[0]).split('.')[0]
    imgname = patchname.split('_')[0]
    if imgname not in img_list:
        indx = unfiltered_img_list.index(imgname)
        imh, imw = unfiltered_img_hw_list[indx]
        img_list.append(imgname)
        img_sz_list.append((imh, imw))
    patch_list.append(patchname)

        
    
patch_name_list = list(pred_dict.keys())
num_classes = 6
patch_h, patch_w = 300., 300.

print('get image level 1-distributions...')

dst = 'ps_valsets/'+dbname+'_'+cnn+'_valset_softmax_1_region_avg_dist.txt'
if not os.path.isdir(os.path.dirname(dst)):
    os.makedirs(os.path.dirname(dst))
    
f = open(dst, 'w')
# create 1-dist per img
for imgname in img_list:
    patches_of_this_img = [patchname for patchname in patch_name_list if imgname in patchname]   
    if len(patches_of_this_img) == 0:
        continue      
    class_dist = np.array([0.] * num_classes)
    for patchname in patches_of_this_img:
        softmax = pred_dict[patchname][0]
        class_dist += np.array(softmax) # elementwise sum using numpy array
        gt = pred_dict[patchname][1] # same for all patches of the image
    avg_class_dist = [float(x) / len(patches_of_this_img) for x in class_dist]
    f.write('{}, {}, {}\n'.format(imgname, gt, ', '.join([str(x)for x in avg_class_dist])))
        
f.close()

print('get image level 4-distributions...')

dst = 'ps_valsets/'+dbname+'_'+cnn+'_valset_softmax_4_region_avg_dist.txt'
f = open(dst, 'w')
# create 4-dist per img 
# region: (row, col)
# region1: (0,0)   <= r1 <= (600,250)
# region2: (0,250) <= r2 <= (600,500) 
# region3: (600,0) <= r3 <= (1200,250) 
# region4: (600,250) <= r3 <= (1200,500) 
for imgname in img_list:
    indx = img_list.index(imgname)
    im_h, im_w = img_sz_list[indx]
    patches_of_this_img = [patchname for patchname in patch_name_list if imgname in patchname]    
    if len(patches_of_this_img) == 0:
        continue     
#    print('len(patches_of_this_img)={}'.format(len(patches_of_this_img)))
    class_dist_r1 = np.array([0.] * num_classes)
    class_dist_r2 = np.array([0.] * num_classes)
    class_dist_r3 = np.array([0.] * num_classes)
    class_dist_r4 = np.array([0.] * num_classes)
    r1_patch_count, r2_patch_count, r3_patch_count, r4_patch_count = 0, 0, 0, 0    
    for patchname in patches_of_this_img:
        top = int(patchname.split('_')[-2])
        left = int(patchname.split('_')[-1])        
        try:
            indx = patch_list.index(patchname)   
        except ValueError:
            continue # found a single patch not found in patch_list for unknown error
#        patch_h, patch_w = patch_sz_list[indx]
        centroid_x = top + patch_h / 2.
        centroid_y = left + patch_w / 2.
        softmax = pred_dict[patchname][0]
#        print('top={}, left={}, c_x={},c_y={}'.format(top,left, centroid_x, centroid_y))
#        print('norm_class_dist={}'.format(str(norm_class_dist)))
        if centroid_x >= 0 and centroid_x < im_h / 2. and centroid_y >= 0 and centroid_y < im_w / 2.:
            class_dist_r1 += np.array(softmax) # elementwise sum using numpy array
            r1_patch_count +=1            
        if centroid_x >= 0 and centroid_x < im_h / 2. and centroid_y >= im_w / 2. and centroid_y <= im_w:
            class_dist_r2 += np.array(softmax) # elementwise sum using numpy array
            r2_patch_count +=1
        if centroid_x >= im_h / 2. and centroid_x <= im_h and centroid_y >= 0 and centroid_y < im_w / 2.:
            class_dist_r3 += np.array(softmax) # elementwise sum using numpy array
            r3_patch_count +=1            
        if centroid_x >= im_h / 2. and centroid_x <= im_h and centroid_y >= im_w / 2. and centroid_y <=im_w:
            class_dist_r4 += np.array(softmax) # elementwise sum using numpy array
            r4_patch_count +=1            
        gt = pred_dict[patchname][1] # same for all patches of the image
#    print('sum_r1={}, sum_of_norm_class_dist_r1={}'.format(sum_r1, str(sum_of_norm_class_dist_r1)))
#    exit(0)
    avg_class_dist_r1 = [float(x) / r1_patch_count if r1_patch_count > 0 else 0 for x in class_dist_r1]
    avg_class_dist_r2 = [float(x) / r2_patch_count if r2_patch_count > 0 else 0 for x in class_dist_r2]
    avg_class_dist_r3 = [float(x) / r3_patch_count if r3_patch_count > 0 else 0 for x in class_dist_r3]
    avg_class_dist_r4 = [float(x) / r4_patch_count if r4_patch_count > 0 else 0 for x in class_dist_r4]
    #concatenate
    avg_class_dist = avg_class_dist_r1 + avg_class_dist_r2 + avg_class_dist_r3 + avg_class_dist_r4
    f.write('{}, {}, {}\n'.format(imgname, gt, ', '.join([str(x)for x in avg_class_dist])))
f.close()

print('get image level 9-distributions...')
    
dst = 'ps_valsets/'+dbname+'_'+cnn+'_valset_softmax_9_region_avg_dist.txt'
f = open(dst, 'w')
# create 4-dist per img 
# region: (row, col)
#region1: (0,0)   <= r1 <= (600,250)
# region2: (0,250) <= r2 <= (600,500) 
# region3: (600,0) <= r3 <= (1200,250) 
# region4: (600,250) <= r3 <= (1200,500) 
for imgname in img_list:
    indx = img_list.index(imgname)
    im_h, im_w = img_sz_list[indx]    
    patches_of_this_img = [patchname for patchname in patch_name_list if imgname in patchname]    
    if len(patches_of_this_img) == 0:
        continue     
    class_dist_r1 = np.array([0.] * num_classes)
    class_dist_r2 = np.array([0.] * num_classes)
    class_dist_r3 = np.array([0.] * num_classes)
    class_dist_r4 = np.array([0.] * num_classes)
    class_dist_r5 = np.array([0.] * num_classes)
    class_dist_r6 = np.array([0.] * num_classes)
    class_dist_r7 = np.array([0.] * num_classes)
    class_dist_r8 = np.array([0.] * num_classes)
    class_dist_r9 = np.array([0.] * num_classes)                    
    r1_patch_count, r2_patch_count, r3_patch_count, r4_patch_count = 0, 0, 0, 0
    r5_patch_count, r6_patch_count, r7_patch_count, r8_patch_count, r9_patch_count = 0, 0, 0, 0, 0                                
    for patchname in patches_of_this_img:
        top = int(patchname.split('_')[-2])
        left = int(patchname.split('_')[-1]) 
        try:
            indx = patch_list.index(patchname)   
        except ValueError:
            continue # found a single patch not found in patch_list for uknown error
#        patch_h, patch_w = patch_sz_list[indx]
        centroid_x = top + patch_h / 2.
        centroid_y = left + patch_w / 2.
        softmax = pred_dict[patchname][0]
        if centroid_x >= 0 and centroid_x < im_h/3. and centroid_y >= 0 and centroid_y <im_w/3.:
            class_dist_r1 += np.array(softmax) # elementwise sum using numpy array
            r1_patch_count +=1            
        if centroid_x >= 0 and centroid_x < im_h/3. and centroid_y >= im_w/3. and centroid_y <2*(im_w/3.):
            class_dist_r2 += np.array(softmax) # elementwise sum using numpy array
            r2_patch_count +=1            
        if centroid_x >= 0 and centroid_x < im_h/3. and centroid_y >= 2*(im_w/3.) and centroid_y <=im_w:
            class_dist_r3 += np.array(softmax) # elementwise sum using numpy array
            r3_patch_count +=1            
        if centroid_x >= im_h/3. and centroid_x < 2*(im_h/3.) and centroid_y >= 0 and centroid_y <im_w/3.:
            class_dist_r4 += np.array(softmax) # elementwise sum using numpy array
            r4_patch_count +=1            
        if centroid_x >= im_h/3. and centroid_x < 2*(im_h/3.) and centroid_y >= im_w/3. and centroid_y <2*(im_w/3.):
            class_dist_r5 += np.array(softmax) # elementwise sum using numpy array
            r5_patch_count +=1            
        if centroid_x >= im_h/3. and centroid_x < 2*(im_h/3.) and centroid_y >= 2*(im_w/3.) and centroid_y <=im_w:
            class_dist_r6 += np.array(softmax) # elementwise sum using numpy array
            r6_patch_count +=1            
        if centroid_x >= 2*(im_h/3.) and centroid_x < im_h and centroid_y >= 0 and centroid_y <im_w/3.:
            class_dist_r7 += np.array(softmax) # elementwise sum using numpy array
            r7_patch_count +=1            
        if centroid_x >= 2*(im_h/3.) and centroid_x < im_h and centroid_y >= im_w/3. and centroid_y <2*(im_w/3.):
            class_dist_r8 += np.array(softmax) # elementwise sum using numpy array
            r8_patch_count +=1            
        if centroid_x >= 2*(im_h/3.) and centroid_x < im_h and centroid_y >= 2*(im_w/3.) and centroid_y <=im_w:
            class_dist_r9 += np.array(softmax) # elementwise sum using numpy array
            r9_patch_count +=1            
        gt = pred_dict[patchname][1] # same for all patches of the image
    avg_class_dist_r1 = [float(x) / r1_patch_count if r1_patch_count > 0 else 0 for x in class_dist_r1]
    avg_class_dist_r2 = [float(x) / r2_patch_count if r2_patch_count > 0 else 0 for x in class_dist_r2]
    avg_class_dist_r3 = [float(x) / r3_patch_count if r3_patch_count > 0 else 0 for x in class_dist_r3]
    avg_class_dist_r4 = [float(x) / r4_patch_count if r4_patch_count > 0 else 0 for x in class_dist_r4]
    avg_class_dist_r5 = [float(x) / r5_patch_count if r5_patch_count > 0 else 0 for x in class_dist_r5]
    avg_class_dist_r6 = [float(x) / r6_patch_count if r6_patch_count > 0 else 0 for x in class_dist_r6]
    avg_class_dist_r7 = [float(x) / r7_patch_count if r7_patch_count > 0 else 0 for x in class_dist_r7]
    avg_class_dist_r8 = [float(x) / r8_patch_count if r8_patch_count > 0 else 0 for x in class_dist_r8]
    avg_class_dist_r9 = [float(x) / r9_patch_count if r9_patch_count > 0 else 0 for x in class_dist_r9]
    # concatenate
    avg_class_dist = avg_class_dist_r1 + avg_class_dist_r2 + avg_class_dist_r3 + \
                      avg_class_dist_r4 + avg_class_dist_r5 + avg_class_dist_r6 + \
                      avg_class_dist_r7 + avg_class_dist_r8 + avg_class_dist_r9
    f.write('{}, {}, {}\n'.format(imgname, gt, ', '.join([str(x)for x in avg_class_dist])))
f.close()
        
        
        
    
