#import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # doctest: +SKIP
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import sys

# create confusion matrices for patch level prediction 
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
#            print(type(cm[i,j]))
            if isinstance(cm[i,j], np.float64):
                cell = "%{0}.3f".format(columnwidth) % cm[i, j]
            else:
                cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
        
dbname = 'gbmlgg_300x300_adaptive_ppi_gridrs1'

comb_id_list = {
    'resnet50_vgg16_rf_1': [7, 42, 47, 5, 13, 41, 55, 56, 1, 32, 35, 37, 50, 52],
    'vgg16_vgg16_rf_1': [21, 29, 101, 10, 11, 14, 15, 18, 22, 24, 5, 89, 1, 4, 7, 12, 13, 16, 19, 23]
}

classes = ['0', '1', '2', '3', '4', '5']



for key in comb_id_list:
    
    parts = key.split('_')

    cnn1 = parts[0]
    cnn2 = parts[1]
    clfname = parts[2]
    dist_type = parts[3] 
    cid_list = comb_id_list[cnn1+'_'+cnn2+'_'+clfname+'_'+dist_type]
    c = 0
    y_score_sum = []
    for comb_id in cid_list:    
        npzfile = np.load('ps_region/'+cnn1+'_'+cnn2+'_'+clfname+'/'+dbname+'_'+cnn1+'_'+cnn2+'_'+dist_type+'_dist_per_image_'+clfname+'_ps'+str(comb_id)+'.npz')
    #            img_list=img_list, y_test=y_test, y_pred=y_pred, y_score=y_score)
        y_test = npzfile['y_test']
        y_pred = npzfile['y_pred']
        y_score = npzfile['y_score']
        if len(y_score_sum) == 0:
            y_score_sum = y_score
        else:
            y_score_sum += y_score
            
        c +=1
    #        break
        y_score_avg = y_score_sum / c
        y_pred_avg = np.argmax(y_score_avg, axis=1)                    
         
        yt_mtx = np.zeros((len(y_test), len(classes))) # size=(n_samples, n_classes)
        ys_mtx = np.zeros((len(y_score_avg), len(classes))) # size=(n_samples, n_classes)

        for i in range(yt_mtx.shape[0]):
            yt_mtx[i, y_test[i]] = 1
            ys_mtx[i] = y_score_avg[i]

        acc = accuracy_score(y_test, y_pred_avg)
        ap_list_macro = average_precision_score(yt_mtx, ys_mtx, 'macro')
        ap_list_micro = average_precision_score(yt_mtx, ys_mtx, 'micro')
        ap_list_wgt = average_precision_score(yt_mtx, ys_mtx, 'weighted')
        ap_list_smpls = average_precision_score(yt_mtx, ys_mtx, 'samples')
        mAP_macro = np.mean(ap_list_macro)
        mAP_micro = np.mean(ap_list_micro)
        mAP_wgt = np.mean(ap_list_wgt)
        mAP_smpls = np.mean(ap_list_smpls)

        print('Val set {}: {}-ensemble accuracy: {:.4f}, mAP(macro): {:.4f}, mAP(micro): {:.4f}, mAP(weighted): {:.4f}, mAP(samples): {:.4f}'.format(key.upper(), c, acc, mAP_macro, mAP_micro, mAP_wgt, mAP_smpls))  
    print()     
