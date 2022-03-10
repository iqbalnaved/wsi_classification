import numpy as np
import os
import matplotlib.pyplot as plt

src = 'GBMLGG_test.txt'
f = open(src)
content = f.readlines()
f.close()

img_dict = {}
k = 1
for line in content:
    parts = line.strip().split(',')
    fpath = parts[0]
    width = int(parts[1].strip(' ('))
    height = int(parts[2].strip(' )'))
    area = width * height
    fname = os.path.basename(fpath).split('.')[0]
    img_dict[fname] = area
    print('{}/{} {} {}'.format(k, len(content), fname, area))
    k +=1

src = 'ps_region/vgg16_vgg16_rf/gbmlgg_300x300_100ppi_rs1_vgg16_vgg16_9_dist_per_image_rf_ps99.npz'
npzfile = np.load(src)
test_img_list = npzfile['img_list']
y_true = npzfile['y_test']
y_pred = npzfile['y_pred']

area_list, label_list = [], []
for i in range(len(test_img_list)):
    imgname = test_img_list[i]
    area = img_dict[imgname]
    label =  int(y_true[i] == y_pred[i])
    area_list.append(area)
    label_list.append(label)
    
n, bins, patches = plt.hist(area_list, bins='auto', edgecolor = "black")  # arguments are passed to np.histogram

plt.xlabel('Image area')
plt.ylabel('No. of images')
plt.title("Frequency distribution by Image Area")
#Text(0.5, 1.0, "Histogram with 'auto' bins")
plt.show()    

bin_accuracy = []
for b in range(len(bins)-1):

    left_edge = bins[b]
    right_edge = bins[b+1]
            
    indices = [i for i, x in enumerate(area_list) if x < right_edge and x >= left_edge]
    correct_freq = sum([label_list[x] for x in indices])
    bin_accuracy.append(correct_freq)    

plt.plot(list(range(1,11)), bin_accuracy,  marker='o')
plt.xticks(list(range(1,11)))
plt.xlabel('Image area bins')
plt.ylabel('No. of correct labels')
plt.title("Frequency distribution of Correct Labels by Image area.")
plt.show()


