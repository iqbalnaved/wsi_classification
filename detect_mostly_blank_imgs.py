import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import os, glob, shutil


src = '/media/adjeroh/10TBHD/DATABASES/Cancer_Datasets/nonblanks/' #TCGA-02-0004-01Z-00-DX1_0_3600.png'
dst = '/media/adjeroh/10TBHD/DATABASES/Cancer_Datasets/mostlyblank/'

fplist = glob.glob(os.path.join(src, '*.png'))
i=0
total_area_dict = {}
for fp in fplist:
    img = cv2.imread(fp,0)
    binary_img = cv2.Canny(img,100,200) # edges

#    plt.subplot(121),plt.imshow(img,cmap = 'gray')
#    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#    plt.show()

    blobs = measure.label(binary_img, connectivity=1)
    props = measure.regionprops(blobs)

    if not props:
#        print('{}/{}: empty props: mostly blank'.format(i+1, len(fplist)))
        total_area = 0
    else:
        total_area = sum([ele.area for ele in props])
    print('{}/{}: total_area={}'.format(i+1, len(fplist), total_area))    
    total_area_dict[os.path.basename(fp)] = total_area
#    area = [ele.area for ele in props]
#    largest_blob_ind = np.argmax(area)
#    largest_blob_label = props[largest_blob_ind].label
    i = i + 1

area_list = total_area_dict.values()

plt.xlim([min(area_list)-5, max(area_list)+5])
plt.title('Histogram of total area in patches')
plt.xlabel('Area')
plt.ylabel('Frequency')
n, bins, patches = plt.hist(area_list, bins=20,alpha=0.5) 
plt.savefig('conn_area_hist.png')
plt.show()

dst = '/media/adjeroh/10TBHD/DATABASES/Cancer_Datasets/mostlyblank'
bc = [0] * len(bins)
sample_per_bin = 25
for filename in total_area_dict:
    for b in range(len(bins)):
        if not os.path.isdir(os.path.join(dst, 'bin_' + str(b))):
            os.makedirs(os.path.join(dst, 'bin_' + str(b)))
        if b == 0:
            if total_area_dict[filename] <= bins[b] and bc[b] <= sample_per_bin: # taking 25 samples/bin
                shutil.copyfile(os.path.join(src, filename), os.path.join(dst, 'bin_' + str(b), filename))
                bc[b] = bc[b] + 1
                print('smpl cnt={}/{}, {}, area={}, bin={}'.format(bc[b], sample_per_bin, filename, total_area_dict[filename], b))
        else:
            if total_area_dict[filename] > bins[b-1] and total_area_dict[filename] <= bins[b] and bc[b] <= sample_per_bin: # taking 25 samples/bin
                shutil.copyfile(os.path.join(src, filename), os.path.join(dst, 'bin_' + str(b), filename))
                bc[b] = bc[b] + 1
                print('smpl cnt={}/{}, {}, area={}, bin={}'.format(bc[b], sample_per_bin, filename, total_area_dict[filename], b))
        

#plt.xlim([min(area_list)-5, max(area_list)+5])
plt.title('Edges of each bin')
plt.xlabel('Bin')
plt.ylabel('Total area')
plt.bar(range(len(bins)), bins) 
plt.savefig('bin_area_plot.png')
plt.show()
#for b in range(len(bins)):
#    count = 0
#    for value in area_list:
#        if value <= bins[b]:
#            count = count + 1        
#    print('values <={}: {:.4f}'.format(bins[b], count))        
#    
#    
#dst = './area_bins'    

#for fp in fplist:
#    val =  area_dict[os.path.basename(fp)]
#    for i in range(len(area_list)):
#        if 

#    

