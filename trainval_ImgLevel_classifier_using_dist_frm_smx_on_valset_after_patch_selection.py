# training on one distribution per image without patch selection
# worfkflow
# patches -> cnn -> features -> svm -> patch lvl pred -> patch selection -> 
# img lvl histogram -> svm -> final pred
# run after
# create_trainset_imglvl_dist_after_patch_selection.py
# create_valset_imglvl_dist.py

import pickle
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import os, glob

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # doctest: +SKIP
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import random
import sys
from joblib import dump
        
# create confusion matrices for patch level prediction 
def fprint_cm(cm, labels, fptr, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    fptr.write("    " + empty_cell)
    for label in labels:
        fptr.write("%{0}s".format(columnwidth) % label)
    fptr.write('\n')
    # Print rows
    for i, label1 in enumerate(labels):
        fptr.write("    %{0}s".format(columnwidth) % label1)
        for j in range(len(labels)):
#            print(type(cm[i,j]))
            if isinstance(cm[i,j], np.float64):
                cell = " %{0}.3f ".format(columnwidth) % cm[i, j]
            else:
                cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            fptr.write(cell)
        fptr.write('\n')

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

classes = ['0', '1', '2', '3', '4', '5']

#cnn1 = 'resnet50'
#cnn2 = 'resnet50'

#cnn1 = 'vgg16'
#cnn2 = 'vgg16'

#cnn1 = 'inceptionv3'
#cnn2 = 'resnet50'

#cnn1 = 'vgg16'
#cnn2 = 'resnet50'

cnn1 = sys.argv[1] #'vgg19bn'
cnn2 = sys.argv[2] #'resnet50'
clfname = sys.argv[3] # 'rf'
#rf_sz = int(sys.argv[4]) # number of decisin trees in RF
dbname = 'gbmlgg_300x300_adaptive_ppi_gridrs1'

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



print(cnn1+'_'+cnn2+'_'+clfname)

for dist_type in ['1', '4', '9']: 

    src = 'ps_valsets/'+dbname+'_'+cnn2+'_valset_softmax_'+dist_type+'_region_avg_dist.txt'

    f = open(src)
    content = f.readlines()
    f.close()

    img_list, X_test,y_test = [], [],[]
    for line in content:
        parts = line.strip().split(',')
        gt = int(parts[1])
        softmax = [float(x) for x in parts[2:]]
        X_test.append(softmax)  
        y_test.append(gt)
        img_list.append(parts[0])



    dst = 'ps_results/'+cnn1+'_'+cnn2+'_'+clfname+'/'+dbname+'_'+cnn1+'_'+cnn2+'_'+clfname+'_valset_softmax_'+dist_type+'_region_avg_dist_prediction_after_ps.txt'
    if not os.path.isdir(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
        
    fp = open(dst, 'w')

#    num_combinations = 125 # 5**5

    for ps_num in comb_id_list[cnn1+'_'+cnn2+'_'+clfname]: #range(1,num_combinations+1):
        # file format
        # imgname, gt, <patch_level prediction histogram (6bin)>
        # one distibution per image
        print('comb_id={} valset prediction after patch selection, dist_type={}'.format(ps_num, dist_type))

        fp.write('trainset patch selection comb id:{}\n'.format(ps_num))    
        
        src = 'ps_trainsets/'+cnn1+'_'+cnn2+'_'+clfname+'/'+dbname+'_'+cnn1+'_'+cnn2+'_trainset_'+dist_type+'_dist_per_img_patch_selection'+str(ps_num)+'.txt'
        f = open(src)
        content = f.readlines()
        f.close()

        X_train,y_train = [],[]
        for line in content:
            parts = line.strip().split(',')
            gt = int(parts[1])
            hist = [float(x) for x in parts[2:]]
            X_train.append(hist)  
            y_train.append(gt)

        print('size(X_train)={},size(y_train)={}'.format(len(X_train), len(y_train)))
        #####################################################################        
        #print('grid searching optimal SVM params...')
        if clfname == 'svm':
            rand_idx = random.sample(range(len(X_train)), 200)
            gX =  [X_train[i] for i in rand_idx]
            gy = [y_train[i] for i in rand_idx]

            parameters = {'kernel': ['rbf'], 'C':[0.0001, 0.001, 0.1, 1, 10, 100, 1e3]}
            svc = svm.SVC()
            clf = GridSearchCV(svc, parameters)
            clf.fit(gX, gy)

            #sorted(clf.cv_results_.keys())
            print(clf.best_params_) 

            C_val = clf.best_params_['C'] 
            
            clf = svm.SVC(C_val)
        #####################################################################        
        # train and test with whole trainset
        elif clfname == 'rf':
            clf = RandomForestClassifier()
#            clf = RandomForestClassifier(n_estimators=rf_sz)
        #####################################################################  
        elif clfname == 'lgrg':
            clf = LogisticRegression(max_iter=1500)
        #####################################################################  
        elif clfname == 'mlp':
            clf = MLPClassifier(random_state=1, max_iter=1000)

        #####################################################################  
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)

        yt_mtx = np.zeros((len(y_test), len(classes))) # size=(n_samples, n_classes)
        ys_mtx = np.zeros((len(y_score), len(classes))) # size=(n_samples, n_classes)

        for i in range(yt_mtx.shape[0]):
            yt_mtx[i, y_test[i]] = 1
            ys_mtx[i] = y_score[i]

        acc = accuracy_score(y_test, y_pred)
        ap_list_macro = average_precision_score(yt_mtx, ys_mtx, 'macro')
        ap_list_micro = average_precision_score(yt_mtx, ys_mtx, 'micro')
        ap_list_wgt = average_precision_score(yt_mtx, ys_mtx, 'weighted')
        ap_list_smpls = average_precision_score(yt_mtx, ys_mtx, 'samples')
        mAP_macro = np.mean(ap_list_macro)
        mAP_micro = np.mean(ap_list_micro)
        mAP_wgt = np.mean(ap_list_wgt)
        mAP_smpls = np.mean(ap_list_smpls)

        print('Validation set {} accuracy: {:.4f}, mAP(macro): {:.4f}, mAP(micro): {:.4f}, mAP(weighted): {:.4f}, mAP(samples): {:.4f}'.format(clfname.upper(), acc, mAP_macro, mAP_micro, mAP_wgt, mAP_smpls))
        fp.write('Validation set {} accuracy: {:.4f}, mAP(macro): {:.4f}, mAP(micro): {:.4f}, mAP(weighted): {:.4f}, mAP(samples): {:.4f}'.format(clfname.upper(), acc, mAP_macro, mAP_micro, mAP_wgt, mAP_smpls))
        fp.write('confusion matrix:\n')
        print('confusion matrix:\n')
        cm = confusion_matrix(y_test, y_pred)
        fprint_cm(cm, classes, fp, hide_zeroes=False, hide_diagonal=False, hide_threshold=None)
        print_cm(cm, classes)
        fp.write('confusion matrix (%):\n')
        print('confusion matrix (%):\n')
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        fprint_cm(cm, classes, fp, hide_zeroes=False, hide_diagonal=False, hide_threshold=None)
        print_cm(cm, classes)
    #    plot_confusion_matrix(clf, X_test, y_test)  # doctest: +SKIP
    #    plt.savefig('raildata_300x300_60ppi_one_dist_per_image_svm_confusion_matrix.png')
    #    plt.show()  # doctest: +SKIP
        if not os.path.isdir('ps_region/'+cnn1+'_'+cnn2+'_'+clfname+'/'):
            os.makedirs('ps_region/'+cnn1+'_'+cnn2+'_'+clfname+'/')
        np.savez('ps_region/'+cnn1+'_'+cnn2+'_'+clfname+'/'+dbname+'_'+cnn1+'_'+cnn2+'_'+dist_type+'_dist_per_image_'+clfname+'_ps'+str(ps_num)+'.npz', \
            img_list=img_list, y_test=y_test, y_pred=y_pred, y_score=y_score)
        if not os.path.isdir('ps_models/'+cnn1+'_'+cnn2+'_'+clfname+'/'):
            os.makedirs('ps_models/'+cnn1+'_'+cnn2+'_'+clfname+'/')

        dump(clf, 'ps_models/'+cnn1+'_'+cnn2+'_'+clfname+'/'+dbname+'_'+cnn1+'_'+cnn2+'_'+dist_type+'_dist_per_image_'+clfname+'_ps'+str(ps_num)+'.model')

    fp.close()
