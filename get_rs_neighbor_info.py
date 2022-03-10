# for each image name
# find all patches
# neigboring patch search radius is defined in  stages,
# search_radius = Cx sqrt(2), where x = hight or width of patches
# if patch centroid distance <= search_radius, it is a neigbhoring patch
# put it in the neighbor list
# initially value of C=1, we increment it by 1 if the number of neighborhood patchs found < 8
# after incrementing all the new neighborhood patches (that are not in the neighborhood list) 
# found, are appended to the neighborhood list
# we keep incrementing search_radious until C = sqrt(im_w^2+im_h^2) 

import os, glob
import pickle 
import math

patch_height = 300
patch_width = 300

ext = 'png'
factor = 1 # 1, 1/2, 3/4

###################################################################

dbname = 'gbmlgg_300x300_adaptive_ppi_gridrs1'

src = '/largeDataVolume/cancer_project/GBMLGG_train.txt'
f = open(src)
content = f.readlines()
f.close()

#src_v = '/largeDataVolume/cancer_project/GBMLGG_test.txt'
#f = open(src_v)
#val_content = f.readlines()
#f.close()

psrc = '/largeDataVolume/LeHou_GBMLGG_300x300_adaptive_ppi_gridwise_random_1/train'
dirlist = os.listdir(psrc)
#psrc_v = '/largeDataVolume/LeHou_GBMLGG_300x300_100ppi_random_1/test'
#dirlist = os.listdir(src_v) # redundant, same set of dirs

dst = '/largeDataVolume/cancer_project/pickles/'+dbname+'_neighbor_list.pkl'


###################################################################
#content = train_content
#content.extend(val_content)

print('create img_list')
img_list =  []
for line in content:    
    fpath = line.split(',')[0].strip()
    fname = os.path.basename(fpath).split('.')[0]
    sz = line.split(',')[1:]
    sz = [int(x.strip().strip('()')) for x in sz]
    h = sz[0]
    w = sz[1]
    img_list.append((fname,h,w))    



###################################################################

print('create patch_list')
patch_list = []
for dirname in dirlist:
#    fpath_list = glob.glob(os.path.join(psrc_t, dirname, '*.' + ext))
#    fname_list = [os.path.basename(x).split('_')[0] for x in fpath_list]
#    img_list.extend(fname_list)    
#    fpath_list = glob.glob(os.path.join(psrc_v, dirname, '*.' + ext))
#    fname_list = [os.path.basename(x).split('_')[0] for x in fpath_list]
#    img_list.extend(fname_list)    

    ppath_list = glob.glob(os.path.join(psrc, dirname, '*.' + ext))
    fname_list = [os.path.basename(x).split('.')[0] for x in ppath_list]
    patch_list.extend(fname_list)    
#    ppath_list = glob.glob(os.path.join(psrc_v, dirname, '*.' + ext))
#    fname_list = [os.path.basename(x).split('.')[0] for x in ppath_list]
#    patch_list.extend(fname_list)    

print('generating neighbor_dict')
neighbor_dict = {}
k = 1
for imgname, img_h, img_w in img_list:
    # since we have square patches, we take the length of the 
    # hypotenous x*sqrt(2), the centroid distance between two diagnoally 
    # adjacent patches, as the threhsold or search radius.
    patches_of_img = [patchname for patchname in patch_list if imgname in patchname]
    print('{}/{} {} has {} patches'.format(k, len(img_list), imgname, len(patches_of_img)))    
    k += 1   
    C = 1     
    if len(patches_of_img) > 8: # skipping the cleaned image
        while C <= (img_w**2 + img_h**2)**0.5:
            search_radius = C * math.ceil((patch_width * 2**0.5) * factor) 
            for patchname in patches_of_img:
                parts = patchname.split('_')
                top = int(parts[-2])
                left = int(parts[-1])
                centroid = (top + patch_height/2. , left + patch_width/2.)
                for neighborpatchname in patches_of_img:
                    parts = neighborpatchname.split('_')
                    top = int(parts[-2])
                    left = int(parts[-1])
                    neighbor_centroid = (top + patch_height/2. , left + patch_width/2.)
                    d = ((centroid[0] - neighbor_centroid[0])**2  + (centroid[1] - neighbor_centroid[1])**2)**0.5
        #            print('centroid distance: {}'.format(d))
                    if patchname not in neighbor_dict:
                        neighbor_dict[patchname] = []    
                    if d <= search_radius and d > 0 and neighborpatchname not in neighbor_dict[patchname]:
                        neighbor_dict[patchname].append(neighborpatchname)
            if len(neighbor_dict[patchname]) < 8: # Chosen from the idea of 8 adjacent neighbor to be minimum
                C += 1
            else:
                break
        print('search radius:{}, {} has {} neighbors'.format(C, patchname, len(neighbor_dict[patchname])))   
    else:
        for patchname in patches_of_img:
            neighbor_dict[patchname] = patches_of_img
            print('search radius:{}, {} has {} neighbors'.format(-1, patchname, len(neighbor_dict[patchname])))   
print('saving to file..')            

#f = open(psrc)
#for patchname in neighbor_dict:
#    f.write('{}'.format(patchname))
#    for i in range(len(neighbor_dict[patchname])):
#        f.write('{}'.format(neighbor_dict[patchnme][i]))
#    f.write('\n')
#f.close()        
with open(dst, 'wb') as f:
    pickle.dump(neighbor_dict, f, pickle.HIGHEST_PROTOCOL)
        
# with open('obj/' + name + '.pkl', 'rb') as f:
#        return pickle.load(f)        
