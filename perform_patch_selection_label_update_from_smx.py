# using patch level prediction perform patch selection and label update
import os
import glob
from scipy.stats import entropy
import numpy as np
from scipy.spatial import distance
import pickle
from statistics import *
import matplotlib
import matplotlib.pyplot as plt
from sys import exit
import heapq
import copy 
import sys

# const variables
SOFTMAX=0
GT = 1
PRED=2
ENTROPY=3
PROB_DIFF=4
N_MEAN_JSD=5
N_STD_JSD=6

#out =  [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]

#H = entropy(out)

#max_prob = max(out)
#sec_max_prob =  sorted(set(out))[-2]

###
#out = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
#out_n1 = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
#out_n2 = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
#out_n3 = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
#out_n4 = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
#out_n5 = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
#out_n6 = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
#out_n7 = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
#out_n8 = [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]

## 
#dist1 = distance.jensenshannon(out, out1)
##...
#dist8 = distance.jensenshannon(out, out8)

#mean_dist = np.mean([dst1,...,dist8])
#std_dist = np.std([dst1,...,dst8])

#############################################################

def neighbor_coherence(pred_dict, filename, neighbor_dict):
# input params
#content: contains file content in format <file-name>, <softmax-output1>, <softmax-output2>, ..., <softmax-opututN)
#filename: filename of center patch
# cp_softmax_output: center patch softmax output
# output params
# neighbor_mean_jsd: mean of jensen-shannon distance from center patch softmax distribution to its 8 neighborhoods
# neighbor_std_jsd: std.dev. of jensen-shannon distance from center patch softmax distribution to its 8 neighborhoods   
    dist_list = []
    cp_softmax_output = pred_dict[filename][SOFTMAX]
    for neighborpatchname in neighbor_dict[filename]:
        softmax_output = pred_dict[neighborpatchname][SOFTMAX]
        dist = distance.jensenshannon(cp_softmax_output, softmax_output)
        dist_list.append(dist)
    neighbor_mean_dist = np.mean(dist_list)
    neighbor_std_dist = np.std(dist_list)
    
    return (neighbor_mean_dist, neighbor_std_dist)
     
    
def get_complex_neighbor_count(filename, pred_dict, entropy_thld, prob_diff_thld, neighbor_dict):

    complex_count = 0
    for neighborpatchname in neighbor_dict[filename]:
        patch_entropy = pred_dict[neighborpatchname][ENTROPY]
        patch_prob_diff = pred_dict[neighborpatchname][PROB_DIFF]
#        print('patch_entropy={:.4f}, entropy_thld={:.4f}, patch_prob_diff={:.4f}, prob_diff_thld={:.4f}'.format(patch_entropy, entropy_thld, patch_prob_diff, prob_diff_thld))
        if patch_entropy >= entropy_thld or patch_prob_diff <= prob_diff_thld:        
            complex_count = complex_count + 1

    return complex_count    


def get_dominant_class(pred_dict, patchname, neighbor_dict, d_class_thld):
#find the class with highest prob_diff among all neighobring patches (including center)

    freq_class_selection = [0] * len(pred_dict[patchname][SOFTMAX])
    
    label = pred_dict[patchname][SOFTMAX].index(max(pred_dict[patchname][SOFTMAX]))
    freq_class_selection[label] += 1.
    for neighborpatchname in neighbor_dict[patchname]:
        label = pred_dict[neighborpatchname][SOFTMAX].index(max(pred_dict[neighborpatchname][SOFTMAX]))
        freq_class_selection[label] += 1.
        
    dominant_class = freq_class_selection.index(max(freq_class_selection))
    sec_dominant_class = heapq.nlargest(2, range(len(freq_class_selection)), key=freq_class_selection.__getitem__)[1]
    
    dominant_class_prob_list = []
    p = pred_dict[patchname][SOFTMAX][dominant_class]
    dominant_class_prob_list.append(p)
    for neighborpatchname in neighbor_dict[patchname]:
        p = pred_dict[neighborpatchname][SOFTMAX][dominant_class]
        dominant_class_prob_list.append(p)

    sec_dominant_class_prob_list = []
    p = pred_dict[patchname][SOFTMAX][sec_dominant_class]
    sec_dominant_class_prob_list.append(p)
    for neighborpatchname in neighbor_dict[patchname]:
        p = pred_dict[neighborpatchname][SOFTMAX][sec_dominant_class]
        sec_dominant_class_prob_list.append(p)
        
    if max(dominant_class_prob_list) - max(sec_dominant_class_prob_list) >= d_class_thld:
        return dominant_class
    else:
        return -1


def get_max_diff_list(pred_dict, neighbor_dict):
# find max_diff, for all images
#for each class, find the class with highest prob_diff among all patches (neighbor and center)
#    max_diff_class_list = []
    max_diff_list = []
    for patchname in pred_dict:
        prob_list = []    
        prob_list += pred_dict[patchname][SOFTMAX] # concat
        for neighborpatchname in neighbor_dict[patchname]:
            prob_list += pred_dict[neighborpatchname][SOFTMAX]

        if len(set(prob_list)) == 1: # handling all 0 case
            max_diff = 0
        else:
            max_prob = max(prob_list)
            sec_max_prob = sorted(set(prob_list))[-2]
            max_diff = max_prob - sec_max_prob

        max_diff_list.append(max_diff)
        
    return max_diff_list
    

#def get_norm_neighbor_class_histogram(neighobor_dict, pred_dict, filename, num_classes=8):
## gets a normalized histogram of class predictions by SVM from the neighobring patches
#    class_dist = [0.0] * num_classes
#    for neighborpatchname in neighbor_dict[filename]:
#        pred = pred_dict[neighborpatchname][PRED]
#        class_dist[pred] +=1.
#    sum_class_dist = sum(class_dist)    
#    class_dist = [ float(x) / sum_class_dist for x in class_dist]

#    return class_dist

    
#############################################################    

# discriminative patch selection rule
# For each patch:
#    prob_diff, entrpy = check_patch_complexity(center_patch)
#    if prob_diff <= prob_diff_thld or entrpy >= entropy_thld: 
#       then patch is complex
#       complex_count = 0
#       For all patches in neighborhood:
#           prob_diff, entrpy = check_patch_complexity(neigbhor_patch)
#           if prob_diff <= prob_diff_thld or entrpy >= entropy_thld: 
#           then patch is complex
#           complex_count += 1
#       If complex_count <  half of the patches in the neighborhood (<4):
#           n_mean_jsd, n_std_jsd = spatial coherence analysis(neighbor_patches, neighbor_softmax, center_patch, center_softmax)
#           if n_mean_jsd <= n_mean_jsd_thld or n_std_jsd <= n_std_jsd_thld:
#                # then patch is coherent
#                class_label = get_dominant_class()
#                if class_label < 0:
#                    no dominant class found (discard patch)
#           else: # non-coherent
#               patch is non-discriminative (discarded)
#       else:
#           patch is non-discriminative (discarded)
#    else:
#        class_label = label of max_prob


# pred_dict format 
# pred_dict[patchname] = {[softmax_output], ground_truth, pred_label, entropy, max_prob-sec_max_prob, neighbor_mean_jsd, neighbor_std_jsd}
def patch_selection_and_label_update(ps_pred_dict, entropy_thld, prob_diff_thld, n_mean_jsd_thld, n_std_jsd_thld, d_class_thld, flog):


    nochange, update, discard1, discard2, discard3 = 0, 0, 0, 0, 0
    for key in ps_pred_dict:
        patch_entropy = ps_pred_dict[key][ENTROPY]
        patch_prob_diff = ps_pred_dict[key][PROB_DIFF]
#        print('patch_entropy={}, entropy_thld={}, patch_prob_diff={}, prob_diff_thld={}'.format(patch_entropy,entropy_thld,patch_prob_diff,prob_diff_thld))
        if patch_entropy >= entropy_thld or patch_prob_diff <= prob_diff_thld:  # check if complex patch 
            complex_count = get_complex_neighbor_count(key, ps_pred_dict, entropy_thld, prob_diff_thld, neighbor_dict)    
#            print('complex_count={}, len(neighbor_dict[key])/2.={}'.format(complex_count, len(neighbor_dict[key]) / 2.))    
            if complex_count < len(neighbor_dict[key]) / 2.: # if less than half neighborhood patches are complex, then perform coherance analysis
                n_mean_jsd, n_std_jsd = ps_pred_dict[key][N_MEAN_JSD], ps_pred_dict[key][N_STD_JSD]
#                print('n_mean_jsd={:.4f}, n_mean_jsd_thld={:.4f}, n_std_jsd={:.4f}, n_std_jsd_thld={:.4f}'.format(n_mean_jsd, n_mean_jsd_thld, n_std_jsd, n_std_jsd_thld))
                if n_mean_jsd <= n_mean_jsd_thld or n_std_jsd <= n_std_jsd_thld:
                    class_label = get_dominant_class(ps_pred_dict, key, neighbor_dict, d_class_thld)
                    if class_label != -1:
                        ps_pred_dict[key][PRED] = class_label # update class label with dominant class
                        update = update + 1
                    else:
                        discard1 = discard1 + 1                        
#                       print('class_label={}, max_prob_diff={}'.format(class_label, max_prob_diff))
                else: # non-coherent
                    ps_pred_dict[key][PRED] = -1  # discard patch              
                    discard2 = discard2 + 1                
            else:
                ps_pred_dict[key][PRED] = -1 # discard patch            
                discard3 = discard3 + 1
        else:
    #        pred_dict[key][1] = pred_dict[key][0].index(max(pred_dict[key][0]))
             nochange = nochange + 1


    print('et={:.5f},pdt={:.5f},nmjt={:.5f},nsjt={:.5f},dct={:.5f}'.format(entropy_thld, prob_diff_thld, n_mean_jsd_thld, n_std_jsd_thld, d_class_thld))
    print('unchanged:{}, updated:{}, discard1:{}, discard2:{}, discard3:{}'.format(nochange, update, discard1, discard2, discard3))
    print('unchanged(%):{:.2f}, updated(%):{:.2f}, discard1(%):{:.2f}, discard2(%):{:.2f}, discard3(%):{:.2f}'.format(float(nochange)/len(ps_pred_dict), float(update)/len(ps_pred_dict), float(discard1)/len(ps_pred_dict), float(discard2)/len(ps_pred_dict), float(discard3)/len(ps_pred_dict)))
    flog.write('et={:.5f},pdt={:.5f},nmjt={:.5f},nsjt={:.5f},dct={:.5f}\n'.format(entropy_thld, prob_diff_thld, n_mean_jsd_thld, n_std_jsd_thld, d_class_thld))
    flog.write('unchanged:{}, updated:{}, discard1:{}, discard2:{}, discard3:{}\n'.format(nochange, update, discard1, discard2, discard3))
    flog.write('unchanged(%):{:.2f}, updated(%):{:.2f}, discard1(%):{:.2f}, discard2(%):{:.2f}, discard3(%):{:.2f}\n'.format(float(nochange)/len(ps_pred_dict), float(update)/len(ps_pred_dict), float(discard1)/len(ps_pred_dict), float(discard2)/len(ps_pred_dict), float(discard3)/len(ps_pred_dict)))
    
    return ps_pred_dict   
 


#############################################################
cnn = sys.argv[1] #'vgg19bn' #'vgg16' #'resnet50' #'inceptionv3' # # classifier type smx or svm
suffix = cnn + "_data_aug_lr_rop_epoch50"

dbname = 'gbmlgg_300x300_adaptive_ppi_gridrs1'

print('loading neighborhood dictionary')

src = 'pickles/'+dbname+'_neighbor_list.pkl'
        
with open(src, 'rb') as f:
   neighbor_dict = pickle.load(f)        

print('loading data to dictionary...')
# pred_dict format 
# pred_dict[patchname] = {[softmax_output], ground_truth, pred_label, entropy, max_prob-sec_max_prob, neighbor_mean_jsd, neighbor_std_jsd}
# change: softmax_outout is now calculated by getting the normalized class histogram
# obtained from the patch neighborhood

#src = 'output/trainset_pred_dict.pkl'
#with open(src, 'rb') as f:
#   pred_dict = pickle.load(f)  

# const variables
#SOFTMAX=0
#GT = 1
#PRED=2
#ENTROPY=3
#PROB_DIFF=4
#N_MEAN_JSD=5
#N_STD_JSD=6

# patchname, ground_truth, predicted label, softmax
#src = 'output/Rail_300x300_65ppi_macroArch_inceptionv3_65ppi_lr_rop_epoch50_net_trainset_smx_predictions.txt'
src = 'features/'+dbname+'_macroArch_'+suffix+'_net_trainset_smx_predictions.txt'

f = open(src)
content = f.readlines()
f.close()

pred_dict = {}
for line in content:
    parts = line.strip().split(',')
    filename = parts[0]
    gt = int(parts[1])
    pred_label =  int(parts[2])
    softmax = [float(x) for x in parts[3:]]
    pred_dict[filename] = [softmax, gt, pred_label] # [] keesp a space for normalized histogram

for filename in pred_dict:
    softmax_output = pred_dict[filename][0]
    entrpy = entropy(softmax_output)
    max_prob = max(softmax_output)
    sec_max_prob = sorted(set(softmax_output))[-2]   
    prob_diff = max_prob - sec_max_prob
    pred_dict[filename][0] = softmax_output 
    pred_dict[filename].append(entrpy)
    pred_dict[filename].append(prob_diff) 

for filename in pred_dict:
    n_mean_jsd, n_std_jsd = neighbor_coherence(pred_dict, filename, neighbor_dict)
    pred_dict[filename].append(n_mean_jsd)
    pred_dict[filename].append(n_std_jsd) 

#with open('output/trainset_smx_pred_dict.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump(pred_dict, f)
    
#pscount = 0
#for patchname in pred_dict:
#    if pred_dict[patchname][PRED] is not -1:
#        pscount +=1
#print('pred_dict select count: {}/{}'.format(pscount, len(pred_dict)))

print('done')

#exit()    
#####################################################################
# check stats of dictionary values
# pred_dict[patchname] = {[softmax_output], ground_truth, pred_label, entropy, max_prob-sec_max_prob, neighbor_mean_jsd, neighbor_std_jsd}
#entropy_thld = np.percentile(entropy_list, percentile_vals[i])  # percentile 
#prob_diff_thld = np.percentile(prob_diff_list, percentile_vals[j])  # percentile 
#n_mean_jsd_thld = np.percentile(n_mean_jsd_list, percentile_vals[k])  # percentile 
#n_std_jsd_thld = np.percentile(n_std_jsd_list, percentile_vals[l])  # percentile 
#d_class_thld = np.percentile(max_diff_list, percentile_vals[m])
max_diff_list = get_max_diff_list(pred_dict, neighbor_dict)
pred_list = list(pred_dict.values())
entropy_list, prob_diff_list = [], []
n_mean_jsd_list, n_std_jsd_list = [],[]

for i in range(len(pred_list)):
    entropy_list.append(pred_list[i][ENTROPY])
    prob_diff_list.append(pred_list[i][PROB_DIFF])
    n_mean_jsd_list.append(pred_list[i][N_MEAN_JSD])
    n_std_jsd_list.append(pred_list[i][N_STD_JSD])

entropy_list = [i for i in entropy_list if i != 0]
prob_diff_list = [i for i in prob_diff_list if i != 0]
n_mean_jsd_list = [i for i in n_mean_jsd_list if i != 0]
n_std_jsd_list = [i for i in n_std_jsd_list if i != 0]
max_diff_list = [i for i in max_diff_list if i != 0]

#print('entropy_list=> min:{}, max:{}, mean:{}, median={}'.format(min(entropy_list), max(entropy_list), mean(entropy_list), median(entropy_list)))
#print('prob_diff_list=> min:{}, max:{}, mean:{}, median={}'.format(min(prob_diff_list), max(prob_diff_list), mean(prob_diff_list), median(prob_diff_list)))
#print('n_mean_jsd_list=> min:{}, max:{}, mean:{}, median={}'.format(min(n_mean_jsd_list), max(n_mean_jsd_list), mean(n_mean_jsd_list), median(n_mean_jsd_list)))
#print('n_std_jsd_list=> min:{}, max:{}, mean:{}, median={}'.format(min(n_std_jsd_list), max(n_std_jsd_list), mean(n_std_jsd_list), median(n_std_jsd_list)))
#print('max_diff_list=> min:{}, max:{}, mean:{}, median={}'.format(min(max_diff_list), max(max_diff_list), mean(max_diff_list), median(max_diff_list)))


#plt.figure(),plt.hist(entropy_list,bins=30)
#plt.ylabel('Frequency'), plt.xlabel('Data'), plt.title("Entropy Histogram");
#plt.savefig("entropy_list_plot.png")
#plt.figure(), plt.hist(prob_diff_list,bins=30)
#plt.ylabel('Frequency'), plt.xlabel('Data'), plt.title("Prob Diff Histogram");
#plt.savefig("prob_diff_list_plot.png")
#plt.figure(), plt.hist(n_mean_jsd_list,bins=30)
#plt.ylabel('Frequency'), plt.xlabel('Data'), plt.title("Norm Mean JSD Histogram");
#plt.savefig("n_mean_jsd_list_plot.png")
#plt.figure(), plt.hist(n_std_jsd_list,bins=30)
#plt.ylabel('Frequency'), plt.xlabel('Data'), plt.title("Norm Std. Dev. JSD Histogram");
#plt.savefig("n_std_jsd_list_plot.png")
#plt.figure(), plt.hist(max_diff_list,bins=30)
#plt.ylabel('Frequency'), plt.xlabel('Data'), plt.title("Max diff Histogram");
#plt.savefig("max_diff_list.png")
#plt.show()

#exit(1)    
#####################################################################
# running patch selection at different thld percentile combinations
print('performing patch selection and label update..')

et_pvals = [90] # [90], [95] , [95]
pdt_pvals = [10] # [5], [10], [5]
nmjt_pvals = [10, 25, 50, 75, 90]
nsjt_pvals = [10, 25, 50, 75, 90]
dct_pvals = [10, 25, 50, 75, 90]
#pred_list = list(pred_dict.values())
#entropy_list = []
#for i in range(len(pred_list)):
#    entropy_list.append(pred_list[i][ENTROPY])
#prob_diff_list = []
#for i in range(len(pred_list)):
#    prob_diff_list.append(pred_list[i][PROB_DIFF])
#n_mean_jsd_list = []
#for i in range(len(pred_list)):
#    n_mean_jsd_list.append(pred_list[i][N_MEAN_JSD])
#n_std_jsd_list = []
#for i in range(len(pred_list)):
#    n_std_jsd_list.append(pred_list[i][N_STD_JSD])
#max_diff_list = get_max_diff_list(pred_dict, neighbor_dict)
flog  = open('log/'+cnn+'_patch_selection.log','w')
c = 1
for i in range(len(et_pvals)):
    for j in range(len(pdt_pvals)):
        for k in range(len(nmjt_pvals)):    
            for l in range(len(nsjt_pvals)):        
                for m in range(len(dct_pvals)):            
                    entropy_thld = np.percentile(entropy_list, et_pvals[i])  # percentile 
                    prob_diff_thld = np.percentile(prob_diff_list, pdt_pvals[j])  # percentile 
                    n_mean_jsd_thld = np.percentile(n_mean_jsd_list, nmjt_pvals[k])  # percentile 
                    n_std_jsd_thld = np.percentile(n_std_jsd_list, nsjt_pvals[l])  # percentile 
                    d_class_thld = np.percentile(max_diff_list, dct_pvals[m])

                    print('pv[i]={},pv[j]={},pv[k]={},pv[l]={},pv[m]={}'.format(et_pvals[i], pdt_pvals[j], nmjt_pvals[k], nsjt_pvals[l], dct_pvals[m]))
                    flog.write('pv[i]={},pv[j]={},pv[k]={},pv[l]={},pv[m]={}\n'.format(et_pvals[i], pdt_pvals[j], nmjt_pvals[k], nsjt_pvals[l], dct_pvals[m]))

                    ps_pred_dict =  copy.deepcopy(pred_dict) # enforce pass by value
                    
#                    pscount = 0
#                    for patchname in ps_pred_dict:
#                        if ps_pred_dict[patchname][PRED] is not -1:
#                            pscount +=1
#                    print('iter={}: Before patch selection: {}  out of {} patches selected'.format(c, pscount, len(ps_pred_dict)))

                    ps_pred_dict = patch_selection_and_label_update(ps_pred_dict, entropy_thld, prob_diff_thld, n_mean_jsd_thld, n_std_jsd_thld, d_class_thld, flog)

                    thld_list = [(entropy_thld, et_pvals[i]), (prob_diff_thld, pdt_pvals[j]), (n_mean_jsd_thld, nmjt_pvals[k]), \
                                (n_std_jsd_thld, nsjt_pvals[l]), (d_class_thld, dct_pvals[m])]
                    
#                    dst = 'patch_selections/'+ctype+'/Rail_macroArch_inceptionv3_65ppi_lr_rop_epoch50_net_trainset_patch_selection' + str(c) + '.pkl'
                    dst = 'patch_selections/'+cnn+'/'+dbname+'_macroArch_'+suffix+'_net_trainset_patch_selection' + str(c) + '.pkl'
                    if not os.path.isdir(os.path.dirname(dst)):
                        os.makedirs(os.path.dirname(dst))
                    pscount = 0
                    for patchname in ps_pred_dict:
                        if ps_pred_dict[patchname][PRED] != -1:
                            pscount +=1
                    print('iter={}: {}  out of {} patches selected'.format(c, pscount, len(ps_pred_dict)))
                    flog.write('iter={}: {}  out of {} patches selected\n\n'.format(c, pscount, len(ps_pred_dict)))

                    c +=1 
                    # Saving the objects:
                    with open(dst, 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump([ps_pred_dict, thld_list], f)
                    print()
#                     Getting back the objects:
#                    with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#                        obj0, obj1, obj2 = pickle.load(f)        
flog.close()        

