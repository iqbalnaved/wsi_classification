# set a 5x5 grid on the image, and from each cell of the grid
#randomly select x ppi and save the selection list. If total_conn_comp < thresh, 
#discard select another randomly, 
#iteratively perform this until x-ppi is achieved.
# we group the patches by cell
# then sample ppc number of patches from each cell where ppc = ppi / gr*gc
# if a cell does not have ppc number of non-blank patches we take as many we can
# before temp runs out of patches. 
# 
import os, glob
import sys
import random
import cv2
from skimage import measure
import numpy as np
#import pandas as pd

def blank_image_check(filepath, threshold):
    img = cv2.imread(filepath,0)
    binary_img = cv2.Canny(img,100,200) # edges
    blobs = measure.label(binary_img, connectivity=1)
    props = measure.regionprops(blobs)
    if not props:
        total_area = 0
    else:
        total_area = sum([ele.area for ele in props])
    if total_area < threshold:
        return True, total_area
    else:
        return False, total_area

def roundup(x):
# round up to the neaest 100th greater than x
    return x if x % 100 == 0 else x + 100 - x % 100   
    
def get_grid_info(imgname, img_w, img_h, grid_row_col):
    
    grid_row = grid_row_col[0]
    grid_col = grid_row_col[1]

    vstride = img_h // grid_row    
    hstride = img_w // grid_col

#     if divisible then add 1 to include 'grid_row' cell edges
#     e.g. if img_h = 10, then range(0,11,2) = [0,2,4,8,10] all number 
#     except 0 indicates right edge of the cell. 
#     if not divisible, we take the first grid_row-1 equally spaced edges, 
#     then take the image boundary as the cell boundary at the last. 
#     e.g. if grid_row=5, img_h = 13, then range(0, 10, 2) = [0,2,4,8] + [13] 
    if img_h % grid_row == 0: 
        cell_row_edges = list(range(0, img_h+1, vstride))
    else:
        cell_row_edges = list(range(0, img_h - (img_h % grid_row), vstride)) + [img_h]

    
    if img_w % grid_col == 0:        
        cell_col_edges = list(range(0, img_w+1, hstride))
    else:
        cell_col_edges = list(range(0, img_w - (img_w % grid_col), hstride)) + [img_w]
    
    cell_positions = (cell_row_edges, cell_col_edges)       
    return cell_positions

def get_cell_no(patchname, cell_positions, patch_sz):
#patchname without extension and directory path   
    patch_w = patch_sz[0]
    patch_h = patch_sz[1]
    top = int(patchname.split('_')[1])
    left = int(patchname.split('_')[2])
    centroid_x = top + patch_h / 2.
    centroid_y = left + patch_w / 2.
    cellno = ()
    cell_row_edges = cell_positions[0]
    cell_col_edges = cell_positions[1]
    for i in range(len(cell_row_edges)-1):
        for j in range(len(cell_col_edges)-1):
            cell_top = cell_row_edges[i]
            cell_bottom = cell_row_edges[i+1]
            cell_left = cell_col_edges[j]
            cell_right = cell_col_edges[j+1]
            if centroid_x >= cell_top and centroid_x <= cell_bottom and centroid_y >= cell_left and centroid_y <= cell_right:
                cellno = (i,j) # cell no determined by topleft coordinate
                break 
        if len(cellno) > 0:
            break
               
    return cellno
  
def group_patches_by_cell(patches_list, cell_positions, patch_sz, grid_row_col):
    
    #initialize
    group_patches = {}
    for i in range(grid_row_col[0]): # 5 rows, 6 endpoints
        for j in range(grid_row_col[1]): # 5 cols, 6 endpoints
            group_patches[(i,j)] = []
            
    for filepath in patches_list:
        patchname = os.path.basename(filepath).split('.')[0]
        top = int(patchname.split('_')[1])
        left = int(patchname.split('_')[2])
        cellno = get_cell_no(patchname, cell_positions, patch_sz)
#        print('{},{}'.format(patchname, cellno))
        group_patches[cellno].append(filepath)
        
    return group_patches

def display_grid(group_patches, grid_row_col):

    grp_total = 0
    for i in range(grid_row_col[0]):
        for j in range(grid_row_col[1]):
            print('{}, '.format(len(group_patches[(i,j)])), end='')
            grp_total += len(group_patches[(i,j)])
        print()
    print('group total = {}'.format(grp_total))   
    print() 


###############################################################################   
dtype = sys.argv[1] # train or test
rs_id = sys.argv[2] # random train/test dataset id
psz = sys.argv[3] # patch size e.g. 300x300
start_idx = int(sys.argv[3])

src = dtype+'set_sample_sizes.npz' # files created using sample_size_plot.py
npzfile = np.load(src)
img_list = npzfile['img_list'].tolist()
#psize_list = npzfile['psize']
ssize_list = npzfile['ssize'].tolist()

threshold = 1339.15 # blank patch threshold
grid_row_col = (5,5)
patch_sz = (300, 300)

src = 'GBMLGG_'+dtype+'.txt'
f = open(src)
content = f.readlines()
f.close()

img_wh_dict = {}
for i in range(len(content)): 

    line = content[i]
    fpath = line.strip().split(',')[0]
    imgname = os.path.basename(fpath).split('.')[0] # TCGA-02-0003-01Z-00-DX2
    img_w = int(line.strip().split(',')[1].strip(' ('))
    img_h = int(line.strip().split(',')[2].strip(' )'))
    img_wh_dict[imgname] = (img_w, img_h)
    
src = 'LeHou_GBMLGG_'+dtype+'.txt'
f = open(src)
content = f.readlines()
f.close()

img_lbl_dict = {}    
for i in range(len(content)): 
    line = content[i]
    fpath = line.strip().split(',')[0]
    imgname = os.path.basename(fpath).split('.')[0] # TCGA-02-0003-01Z-00-DX2
    label = line.strip().split(',')[1].strip()
    img_lbl_dict[imgname] = label
    
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

#previous_random_selections = []
#if rs_id > 1:
#    for i in range(rs_id):
#        csv_file = 'LeHou_GBMLGG_300x300_'+str(ppi)+'ppi_random_' + dtype + '_' + i + '.txt'
#        df = pd.read_csv(csv_file)
#        fpath_list = df.iloc[:, 0].tolist()
#        patch_list = [os.path.basename(fp) for fp in fpath_list]
#        previous_random_selections.extends(patch_list)

## load previous blank detections
#csv_file = 'LeHou_GBMLGG_300x300_100ppi_' + dtype + '_blank_patch_list.txt'
#df = pd.read_csv(csv_file)
#fpath_list = df.iloc[:, 0].tolist()
#patch_list = [os.path.basename(fp) for fp in fpath_list]
#blank_patch_list = patch_list
# 
#csv_file = 'LeHou_GBMLGG_300x300_100ppi_' + dtype + '_blank_img_list.txt'
#df = pd.read_csv(csv_file)
#fpath_list = df.iloc[:, 0].tolist()
#img_list = [os.path.basename(fp) for fp in fpath_list]
#blank_img_list = img_list
if start_idx == 0:
    fmode = 'w'
else:
    fmode = 'a'
    
f = open('LeHou_GBMLGG_300x300_adaptive_ppi_gridwise_random_' + dtype + '_' + rs_id + '.txt', fmode)
#g = open('LeHou_GBMLGG_300x300_' + dtype + '_blank_patch_list.txt', 'a')
#h = open('LeHou_GBMLGG_300x300_' + dtype + '_blank_img_list.txt', 'a')

if rs_id == '1':
    sample_size_factor = 1. # used to create different size training sets
elif rs_id == '2':
    sample_size_factor =  .5 
elif rs_id == '3':
    sample_size_factor =  .25
elif rs_id == '4':    
    sample_size_factor = 2. 

for i in range(start_idx, len(img_wh_dict)): 

    
    imgname = list(img_wh_dict.keys())[i]
    
#    if i != 79: # debug
#        continue

    
    idx = img_list.index(imgname)
    sample_size = round(ssize_list[idx]) # patch / image
    ppi = roundup(sample_size)  * sample_size_factor
    ppc = ppi / (grid_row_col[0]*grid_row_col[1])    # patch / cell

    img_w = img_wh_dict[imgname][0]
    img_h = img_wh_dict[imgname][1]
    cell_positions = get_grid_info(imgname, img_w, img_h, grid_row_col)
#    print(cell_positions)
#    if imgname is in blank_img_list:
#        continue
    label = img_lbl_dict[imgname]
    patches_of_this_img = glob.glob(os.path.join(psrc, label, imgname + '_*.png'))
#    print('no of patches of {} = {}'.format(imgname, len(patches_of_this_img)))

    group_patches = group_patches_by_cell(patches_of_this_img, cell_positions, patch_sz, grid_row_col)
#    display_grid(group_patches, grid_row_col)    
    
    selected_patch_list = []
    blank_patch_count = 0
    selected_cell_patch_count = dict.fromkeys(group_patches, 0) # copy keys from group_patches to new dict and init to 0
    for cellno in group_patches: # cellno is a tuple (i,j)
        cell_patches = group_patches[cellno]
        temp = cell_patches.copy()
        while selected_cell_patch_count[cellno] < ppc: 
            if len(temp) == 0:
#                print('{} out of {} patches of {}-th cell are blank.'.format(blank_patch_count, len(cell_patches), k))
#                if len(patches_of_this_img) == blank_patch_count:
#                    h.write('{}\n'.format(imgname)) # all patches of this image is blank
                break

            rand_idx = random.sample(range(len(temp)), 1)[0]
            #            # check if this patch has been already selected in previous train and test set
            #            if temp[rand_idx] is in previous_random_selections:
            #                continue
            #            # check if this patch is in blank patch list 
            #            if temp[rand_idx] is in blank_patch_list:
            #                continue        
            patchname = os.path.basename(temp[rand_idx]).split('.')[0]
#            if selected_cell_patch_count[cellno] == ppc:
#                break
            isblank, total_area = blank_image_check(temp[rand_idx], threshold)
            if isblank:    
#                g.write('{}, {}\n'.format(temp[rand_idx], total_area)) # blank patch detected
                blank_patch_count +=1 
            else:
                selected_patch_list.append(temp[rand_idx])
                selected_cell_patch_count[cellno] += 1
                f.write('{}, {}\n'.format(temp[rand_idx], label))

            #        print('rand_idx={}, temp[rand_idx]={}'.format(rand_idx, temp[rand_idx]))
            temp.remove(temp[rand_idx]) # remove the patch from temporary patches_of_this_img list, so it doesn't get selected again.
    
    print('{}: {}, ppi={}, ppc={}'.format(i, imgname, ppi, ppc))

#    for r in range(5):
#        for c in range(5):
#            print('{}, '.format(selected_cell_patch_count[(r,c)]), end='')
#        print()
#    print()

    
# count the number of 0-sample cells, then count the number of missing samples , 
# then divide this number with number of cells containing ppc samples
# (ppi - sample_count) / num_of_cells. 
# And, for those cells we add the missing samples using this formula so that 
# the selected random patches are evenly distributed. 
    if len(selected_patch_list) < ppi:
        ppc_cell_count = 0
        for cellno in selected_cell_patch_count:
            if selected_cell_patch_count[cellno] == ppc:
                ppc_cell_count +=1
        if ppc_cell_count == 0: # blank image 
            print('blank image')
            continue
        missing_sample_count = ppi - len(selected_patch_list)
        additional_sample_count = round(missing_sample_count / ppc_cell_count)
        for cellno in group_patches: # cellno is a tuple (i,j)
            if selected_cell_patch_count[cellno] == ppc:
                cell_patches = group_patches[cellno]
                temp = cell_patches.copy()
                while selected_cell_patch_count[cellno] < ppc + additional_sample_count: 
                    if len(temp) == 0:
        #                print('{} out of {} patches of {}-th cell are blank.'.format(blank_patch_count, len(cell_patches), k))
        #                if len(patches_of_this_img) == blank_patch_count:
        #                    h.write('{}\n'.format(imgname)) # all patches of this image is blank
                        break

                    rand_idx = random.sample(range(len(temp)), 1)[0]
                    #            # check if this patch has been already selected in previous train and test set
                    #            if temp[rand_idx] is in previous_random_selections:
                    #                continue
                    #            # check if this patch is in blank patch list 
                    #            if temp[rand_idx] is in blank_patch_list:
                    #                continue        
#                    patchname = os.path.basename(temp[rand_idx]).split('.')[0]
                    if temp[rand_idx] not in selected_patch_list :

                        isblank, total_area = blank_image_check(temp[rand_idx], threshold)
                        if isblank:    
            #                g.write('{}, {}\n'.format(temp[rand_idx], total_area)) # blank patch detected
                            blank_patch_count +=1 
                        else:
                            selected_patch_list.append(temp[rand_idx])
                            selected_cell_patch_count[cellno] += 1
                            f.write('{}, {}\n'.format(temp[rand_idx], label))

                    #        print('rand_idx={}, temp[rand_idx]={}'.format(rand_idx, temp[rand_idx]))
                    temp.remove(temp[rand_idx]) # remove the patch from temporary patches_of_this_img list, so it doesn't get selected again.        
        
    for r in range(5):
        for c in range(5):
            print('{}, '.format(selected_cell_patch_count[(r,c)]), end='')
        print()
    print()                
        
#    break # debug
    
f.close()
#g.close()
#h.close()

    
