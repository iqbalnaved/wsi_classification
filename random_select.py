#randomly select x ppi and save the selection list. If total_conn_comp < thresh, 
#discard select another randomly, 
#iteratively perform this until x-ppi is achieved.

import os, glob
import sys
import random
import cv2
from skimage import measure
import pandas as pd

threshold = 1339.15 # blank patch threshold

dtype = sys.argv[1] # train or test
rs_id = sys.argv[2] # random selection id, each selection should have a unique set (how to check ? TODO)

src = 'LeHou_GBMLGG_'+dtype+'.txt'
f = open(src)
content = f.readlines()
f.close()

psrc = '/largeDataVolume/LeHou_GBMLGG_300x300/' + dtype

###############################################################################
#class_count = {0:0,1:0,2:0,3:0,4:0,5:0}
#for i in range(len(content)):
#    line = content[i]
#    label = int(line.strip().split(',')[1].strip())
#    class_count[label] += 1
#    

#print('{}-set:'.format(dtype))

#for label in class_count:
#    print('class-{}: {}'.format(label, class_count[label]))
    
#train-set:
#class-0: 391
#class-1: 169
#class-2: 146
#class-3: 97
#class-4: 28
#class-5: 11

#test-set:
#class-0: 118
#class-1: 33
#class-2: 37
#class-3: 17
#class-4: 8
#class-5: 4

###############################################################################

previous_random_selections = []
if rs_id > 1:
    for i in range(rs_id):
        csv_file = 'LeHou_GBMLGG_300x300_100ppi_random_' + dtype + '_' + i + '.txt'
        df = pd.read_csv(csv_file)
        fpath_list = df.iloc[:, 0].tolist()
        patch_list = [os.path.basename(fp) for fp in fpath_list]
        previous_random_selections.extends(patch_list)


csv_file = 'LeHou_GBMLGG_300x300_100ppi_' + dtype + '_blank_patch_list.txt'
df = pd.read_csv(csv_file)
fpath_list = df.iloc[:, 0].tolist()
patch_list = [os.path.basename(fp) for fp in fpath_list]
blank_patch_list = patch_list
 
csv_file = 'LeHou_GBMLGG_300x300_100ppi_' + dtype + '_blank_img_list.txt'
df = pd.read_csv(csv_file)
fpath_list = df.iloc[:, 0].tolist()
img_list = [os.path.basename(fp) for fp in fpath_list]
blank_img_list = img_list


ppi = 100

f = open('LeHou_GBMLGG_300x300_100ppi_random_' + dtype + '_' + rs_id + '.txt', 'w')
g = open('LeHou_GBMLGG_300x300_100ppi_' + dtype + '_blank_patch_list.txt', 'a')
h = open('LeHou_GBMLGG_300x300_100ppi_' + dtype + '_blank_img_list.txt', 'a')
for i in range(len(content)): 

    line = content[i]
    fpath = line.strip().split(',')[0]
    filename = os.path.basename(fpath).split('.')[0] # TCGA-02-0003-01Z-00-DX2
    if filename is in blank_img_list:
        continue
    label = line.strip().split(',')[1].strip()

    patches_of_this_img = glob.glob(os.path.join(psrc, label, filename + '_*.png')) # all patches of an image should have the same label.
    selected_patch_count = 0
    temp = patches_of_this_img.copy()
    blank_patch_count = 0
    while selected_patch_count < ppi:

#        print('len(temp)={}'.format(len(temp)))
        if len(temp) == 0:
            print('{} out of {} patches of {} are blank.'.format(blank_patch_count, len(patches_of_this_img), filename))
            if len(patches_of_this_img) == blank_patch_count:
                h.write('{}\n'.format(filename)) # all patches of this image is blank
            break

        rand_idx = random.sample(range(len(temp)), 1)[0]
        # check if this patch has been already selected in previous train and test set
        if temp[rand_idx] is in previous_random_selections:
            continue
        # check if this patch is in blank patch list 
        if temp[rand_idx] is in blank_patch_list:
            continue        
        img = cv2.imread(temp[rand_idx],0)
        binary_img = cv2.Canny(img,100,200) # edges
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            total_area = 0
        else:
            total_area = sum([ele.area for ele in props])
        if total_area < threshold:    
            g.write('{}, {}\n'.format(temp[rand_idx], total_area)) # blank patch detected
            blank_patch_count +=1 
        else:
            selected_patch_count += 1
            print('{}-th img, {}-th patch: {}, {}'.format(i+1, selected_patch_count, temp[rand_idx], label))
            f.write('{}, {}\n'.format(temp[rand_idx], label))

#        print('rand_idx={}, temp[rand_idx]={}'.format(rand_idx, temp[rand_idx]))
        temp.remove(temp[rand_idx]) # remove the patch from temporary patches_of_this_img list, so it doesn't get selected again.

f.close()
g.close()
h.close()

    
