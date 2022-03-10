import pickle
import os, glob
import numpy as np
import sys

# RUN AFTER PATCH SELECTION
# const variables
SOFTMAX=0
GT = 1
PRED=2
ENTROPY=3
PROB_DIFF=4
N_MEAN_JSD=5
N_STD_JSD=6

num_classes = 6
      
src = '/largeDataVolume/cancer_project/GBMLGG_train.txt' #'/media/adjeroh/10TBHD/DATABASES/Rail_Surface_Database/rail_images_trainval/train'
f = open(src)
content = f.readlines()
f.close()
unfiltered_img_list =  [] # image list before applying blank removal 
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
    
src = 'LeHou_GBMLGG_300x300_adaptive_ppi_gridwise_random_train_1.txt' 
f = open(src)
content = f.readlines()
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
        
   
print('loading data to dictionary...')
# pred_dict format 
# pred_dict[patchname] = {[softmax_output], ground_truth, pred_label, entropy, max_prob-sec_max_prob, neighbor_mean_jsd, neighbor_std_jsd}
# change: softmax_outout is now calculated by getting the normalized class histogram
# obtained from the patch neighborhood
patch_h, patch_w = 300. , 300.

cnn1 = sys.argv[1] #'vgg19bn' #'vgg16' #'inceptionv3' #'vgg16' #'resnet50'
cnn2 = sys.argv[2] #'resnet50' #'vgg16' #'resnet50'
clf = sys.argv[3]
dbname = 'gbmlgg_300x300_adaptive_ppi_gridrs1'
suffix = cnn1+'_data_aug_lr_rop_epoch50'

#ctype= 'inceptionv3' #'smx3' # patchlevel classifier 
#ftype = 'dist_from_smx5' #'dist_from_smx3' # generating feature type 
#suffix = ctype+'_65ppi_data_aug_lr_rop_epoch50'
#num_comb = 125
#for i in range(1,num_comb+1): # 5^5 threshold combinations
#comb_id_list = [47, 108, 109, 22, 1, 98, 89, 28, 63, 87, 113, 55, 123, 125, 88, 82, 29, 33, 104, 85, 71, 114, 39, 100, 12, 96, 23] # inceptionv3


#comb_id_list={
#    'resnet50_resnet50_svm': [12,  13,  14,  15,  39, 40, 61, 63, 78, 106, 107, 108, 109, 110], 
#    'resnet50_resnet50_rf': [61], 
#    'resnet50_resnet50_lgrg': [6, 7, 8, 9, 10, 31, 32, 33, 34,  35,  51,  52, 53,  54,  55,  56,  57, 58, 59, 60], 
#    'resnet50_resnet50_mlp': [16,  17,  18,  19,  20, 41,  42,  43,  44,  45,76,  77,  78,  79,  80,111,  112,  113,  114,115],    
#         
#    'inceptionv3_resnet50_svm': [11, 12, 13, 14, 37, 38, 39, 61, 63, 64, 65, 97],   
#    'inceptionv3_resnet50_rf': [29, 55, 56],       
#    'inceptionv3_resnet50_lgrg': [21, 22, 23, 24, 25, 46, 47, 48, 49, 50, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 96, 97, 98, 99, 100, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125],           
#    'inceptionv3_resnet50_mlp': [86, 87, 88, 89, 90],  
#                 
#    'vgg16_resnet50_svm': [51, 53, 54, 55], 
#    'vgg16_resnet50_rf': [44], 
#    'vgg16_resnet50_lgrg': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 101, 102, 103, 104, 105], 
#    'vgg16_resnet50_mlp': [51, 52, 53, 54, 55],          
#       
#    'vgg19bn_resnet50_svm': [52, 53, 54], 
#    'vgg19bn_resnet50_rf': [38, 41, 76, 100, 108, 114],
#    'vgg19bn_resnet50_lgrg': [81, 82, 83, 84, 85],
#    'vgg19bn_resnet50_mlp': [6, 7, 8, 9, 10, 31, 32, 33, 34, 35, 121, 122, 123, 124, 125]            
#}

comb_id_list = {
    'resnet50_vgg16_rf': [1, 2, 5, 7, 13, 19, 21, 22, 32, 35, 37, 39, 41, 42, 47, 50, 51, 52, 55, 56], #58, 63, 74, 76, 78], #79, 80, 81, 84, 87, 88, 92, 102, 111, 115, 116, 118],
    'vgg16_vgg16_rf': [5, 21, 29, 89, 101, 1, 4, 7, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24], #25, 30, 33, 34, 37], #38, 39, 40, 41, 42, 44, 45, 46, 49, 54, 55, 59, 60, 61, 64, 66, 67, 72, 74, 76, 77, 78, 79, 82, 84, 86, 87, 92, 94, 95, 96, 98, 100, 103, 104, 106, 108, 109, 112, 114, 115, 117, 118, 119, 120, 121, 123, 124, 125],
    'inceptionv3_vgg16_rf': [108, 27, 56, 65, 73],
    'vgg19bn_vgg16_rf': [99, 117, 1, 7, 8, 9, 13, 18, 19, 20, 42, 43, 50, 58, 59, 61, 63, 65, 72] #, 76, 77, 79, 84, 86, 88] #, 100, 107, 108, 119, 121, 125]
}

for i in comb_id_list[cnn1+'_'+cnn2+'_'+clf]:    

    # patchname, ground_truth, predicted label
    src = 'patch_selections/'+cnn1+'/'+dbname+'_macroArch_'+suffix+'_net_trainset_patch_selection'+str(i)+'.pkl'

    ##Getting back the objects:
    with open(src, 'rb') as f:  # Python 3: open(..., 'rb')
        selected_patches_dict,_ = pickle.load(f)   

    patch_name_list = list(selected_patches_dict.keys())

    print('combid={} get image list and calculate image level 1-distributions...'.format(i))
    
    dst = 'ps_trainsets/'+cnn1 + '_' + cnn2+'_'+clf+'/'+dbname+'_'+cnn1+'_' +cnn2 +'_trainset_1_dist_per_img_patch_selection'+str(i)+'.txt'
    if not os.path.isdir('ps_trainsets/'+cnn1 + '_' + cnn2+'_'+clf+'/'):
        os.makedirs('ps_trainsets/'+cnn1 + '_' + cnn2+'_'+clf+'/')
    f = open(dst, 'w')
    # create 1-dist per img
    for imgname in img_list:
        patches_of_this_img = [patchname for patchname in patch_name_list if imgname in patchname]  
        if len(patches_of_this_img) == 0:
            print('comid={}, imgname={}'.format(i,imgname))
            continue  
        sum_class_dist = np.array([0.] * num_classes)
        for patchname in patches_of_this_img:
            if selected_patches_dict[patchname][PRED] == -1:
                gt = selected_patches_dict[patchname][GT] # same for all patches of the image
                continue
            class_dist = selected_patches_dict[patchname][SOFTMAX]
            sum_class_dist += np.array(class_dist) # elementwise sum using numpy array
            gt = selected_patches_dict[patchname][GT] # same for all patches of the image
        avg_class_dist = [float(x) / len(patches_of_this_img) for x in sum_class_dist]
        f.write('{}, {}, {}\n'.format(imgname, gt, ', '.join([str(x)for x in avg_class_dist])))            
    f.close()

    print('combid={} get image level 4-distributions...'.format(i))
        
    dst = 'ps_trainsets/'+cnn1 + '_' + cnn2+'_'+clf+'/'+dbname+'_'+cnn1+'_' +cnn2 +'_trainset_4_dist_per_img_patch_selection'+str(i)+'.txt'
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
        sum_class_dist_r1 = np.array([0.] * num_classes)
        sum_class_dist_r2 = np.array([0.] * num_classes)
        sum_class_dist_r3 = np.array([0.] * num_classes)
        sum_class_dist_r4 = np.array([0.] * num_classes)
        r1_patch_count, r2_patch_count, r3_patch_count, r4_patch_count = 0, 0, 0, 0
        for patchname in patches_of_this_img:
            if selected_patches_dict[patchname][PRED] == -1:
                gt = selected_patches_dict[patchname][GT] # same for all patches of the image
                continue
            top = int(patchname.split('_')[-2])
            left = int(patchname.split('_')[-1])     
            indx = patch_list.index(patchname)   
#            patch_h, patch_w = patch_sz_list[indx]
            centroid_x = top + patch_h / 2.
            centroid_y = left + patch_w / 2.
            class_dist = selected_patches_dict[patchname][0]
            if centroid_x >= 0 and centroid_x < im_h / 2. and centroid_y >= 0 and centroid_y < im_w / 2.:
                sum_class_dist_r1 += np.array(class_dist) # elementwise sum using numpy array
                r1_patch_count +=1
            if centroid_x >= 0 and centroid_x < im_h / 2. and centroid_y >= im_w / 2. and centroid_y <= im_w:
                sum_class_dist_r2 += np.array(class_dist) # elementwise sum using numpy array
                r2_patch_count +=1                
            if centroid_x >= im_h / 2. and centroid_x <= im_h and centroid_y >= 0 and centroid_y < im_w / 2.:
                sum_class_dist_r3 += np.array(class_dist) # elementwise sum using numpy array
                r3_patch_count +=1                
            if centroid_x >= im_h / 2. and centroid_x <= im_h and centroid_y >= im_w / 2. and centroid_y <= im_w:
                sum_class_dist_r4 += np.array(class_dist) # elementwise sum using numpy array
                r4_patch_count +=1
            gt = selected_patches_dict[patchname][1] # same for all patches of the image
        avg_class_dist_r1 = [float(x) / r1_patch_count if r1_patch_count > 0 else 0 for x in sum_class_dist_r1]
        avg_class_dist_r2 = [float(x) / r2_patch_count if r2_patch_count > 0 else 0 for x in sum_class_dist_r2]
        avg_class_dist_r3 = [float(x) / r3_patch_count if r3_patch_count > 0 else 0 for x in sum_class_dist_r3]
        avg_class_dist_r4 = [float(x) / r4_patch_count if r4_patch_count > 0 else 0 for x in sum_class_dist_r4]
        #concatenate
        avg_class_dist = avg_class_dist_r1 + avg_class_dist_r2 + avg_class_dist_r3 + avg_class_dist_r4 
        f.write('{}, {}, {}\n'.format(imgname, gt, ', '.join([str(x)for x in avg_class_dist])))
    f.close()

    print('combid={} get image level 9-distributions...'.format(i))
        
    dst = 'ps_trainsets/'+cnn1 + '_' + cnn2+'_'+clf+'/'+dbname+'_'+cnn1+'_' +cnn2 +'_trainset_9_dist_per_img_patch_selection'+str(i)+'.txt'
    f = open(dst, 'w')
    for imgname in img_list:
        indx = img_list.index(imgname)
        im_h, im_w = img_sz_list[indx]  
        patches_of_this_img = [patchname for patchname in patch_name_list if imgname in patchname]  
        if len(patches_of_this_img) == 0:
            continue            
        sum_class_dist_r1 = np.array([0.] * num_classes)
        sum_class_dist_r2 = np.array([0.] * num_classes)
        sum_class_dist_r3 = np.array([0.] * num_classes)
        sum_class_dist_r4 = np.array([0.] * num_classes)
        sum_class_dist_r5 = np.array([0.] * num_classes)
        sum_class_dist_r6 = np.array([0.] * num_classes)
        sum_class_dist_r7 = np.array([0.] * num_classes)
        sum_class_dist_r8 = np.array([0.] * num_classes)
        sum_class_dist_r9 = np.array([0.] * num_classes)    
        r1_patch_count, r2_patch_count, r3_patch_count, r4_patch_count = 0, 0, 0, 0
        r5_patch_count, r6_patch_count, r7_patch_count, r8_patch_count, r9_patch_count = 0, 0, 0, 0, 0                                
        for patchname in patches_of_this_img:
            if selected_patches_dict[patchname][PRED] == -1:
                gt = selected_patches_dict[patchname][GT] # same for all patches of the image
                continue
            top = int(patchname.split('_')[-2])
            left = int(patchname.split('_')[-1]) 
            indx = patch_list.index(patchname)   
#            patch_h, patch_w = patch_sz_list[indx]            
            centroid_x = top + patch_h / 2.
            centroid_y = left + patch_w / 2.
            class_dist = selected_patches_dict[patchname][0]
            if centroid_x >= 0 and centroid_x < im_h/3. and centroid_y >= 0 and centroid_y < im_w/3.:
                sum_class_dist_r1 += np.array(class_dist) # elementwise sum using numpy array
                r1_patch_count +=1
            if centroid_x >= 0 and centroid_x < im_h/3. and centroid_y >= im_w/3. and centroid_y < 2*(im_w/3.):
                sum_class_dist_r2 += np.array(class_dist) # elementwise sum using numpy array
                r2_patch_count +=1
            if centroid_x >= 0 and centroid_x < im_h/3. and centroid_y >= 2*(im_w/3.) and centroid_y <= im_w:
                sum_class_dist_r3 += np.array(class_dist) # elementwise sum using numpy array
                r3_patch_count +=1
            if centroid_x >= im_h/3. and centroid_x < 2*(im_h/3.) and centroid_y >= 0 and centroid_y < im_w/3.:
                sum_class_dist_r4 += np.array(class_dist) # elementwise sum using numpy array
                r4_patch_count +=1
            if centroid_x >= im_h/3. and centroid_x < 2*(im_h/3.) and centroid_y >= im_w/3. and centroid_y < 2*(im_w/3.):
                sum_class_dist_r5 += np.array(class_dist) # elementwise sum using numpy array
                r5_patch_count +=1
            if centroid_x >= im_h/3. and centroid_x < 2*(im_h/3.) and centroid_y >= 2*(im_w/3.) and centroid_y <= im_w/3.:
                sum_class_dist_r6 += np.array(class_dist) # elementwise sum using numpy array
                r6_patch_count +=1
            if centroid_x >= 2*(im_h/3.) and centroid_x < im_h and centroid_y >= 0 and centroid_y < im_w/3.:
                sum_class_dist_r7 += np.array(class_dist) # elementwise sum using numpy array
                r7_patch_count +=1
            if centroid_x >= 2*(im_h/3.) and centroid_x < im_h and centroid_y >= im_w/3. and centroid_y < 2*(im_w/3.):
                sum_class_dist_r8 += np.array(class_dist) # elementwise sum using numpy array
                r8_patch_count +=1
            if centroid_x >= 2*(im_h/3.) and centroid_x < im_h and centroid_y >= 2*(im_w/3.) and centroid_y <= im_w:
                sum_class_dist_r9 += np.array(class_dist) # elementwise sum using numpy array
                r9_patch_count +=1
            gt = selected_patches_dict[patchname][1] # same for all patches of the image
        avg_class_dist_r1 = [float(x) / r1_patch_count if r1_patch_count > 0 else 0 for x in sum_class_dist_r1]
        avg_class_dist_r2 = [float(x) / r2_patch_count if r2_patch_count > 0 else 0 for x in sum_class_dist_r2]
        avg_class_dist_r3 = [float(x) / r3_patch_count if r3_patch_count > 0 else 0 for x in sum_class_dist_r3]
        avg_class_dist_r4 = [float(x) / r4_patch_count if r4_patch_count > 0 else 0 for x in sum_class_dist_r4]
        avg_class_dist_r5 = [float(x) / r5_patch_count if r5_patch_count > 0 else 0 for x in sum_class_dist_r5]
        avg_class_dist_r6 = [float(x) / r6_patch_count if r6_patch_count > 0 else 0 for x in sum_class_dist_r6]
        avg_class_dist_r7 = [float(x) / r7_patch_count if r7_patch_count > 0 else 0 for x in sum_class_dist_r7]
        avg_class_dist_r8 = [float(x) / r8_patch_count if r8_patch_count > 0 else 0 for x in sum_class_dist_r8]
        avg_class_dist_r9 = [float(x) / r9_patch_count if r9_patch_count > 0 else 0 for x in sum_class_dist_r9]
        #concatenate
        avg_class_dist = avg_class_dist_r1 + avg_class_dist_r2 + avg_class_dist_r3 + \
                         avg_class_dist_r4 + avg_class_dist_r5 + avg_class_dist_r6 + \
                         avg_class_dist_r7 + avg_class_dist_r8 + avg_class_dist_r9
     
        f.write('{}, {}, {}\n'.format(imgname, gt, ', '.join([str(x)for x in avg_class_dist])))
    f.close()

                
                
            
