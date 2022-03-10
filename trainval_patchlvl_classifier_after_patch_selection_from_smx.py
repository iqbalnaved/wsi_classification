# takes patch level deep features as input
# trains svm on trainset features (patch level)
# predicts on valset features (patch level)

import pickle
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import os, glob
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # doctest: +SKIP
from sklearn.metrics import plot_confusion_matrix
import numpy as np
from joblib import dump
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

def fprint_cm(cm, labels, fp, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    fp.write("    " + empty_cell)
    for label in labels:
        fp.write(" %{0}s".format(columnwidth) % label)
    fp.write('\n')
    # Print rows
    for i, label1 in enumerate(labels):
        fp.write("    %{0}s".format(columnwidth) % label1)
        for j in range(len(labels)):
#            print(type(cm[i,j]))
            if isinstance(cm[i,j], np.float64):
                cell = " %{0}.3f".format(columnwidth) % cm[i, j]
            else:
                cell = " %{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            fp.write(cell)
        fp.write('\n')

cnn = sys.argv[1] #'vgg19bn'  #'vgg16' #'resnet50' #  'inceptionv3' #       
clfname = sys.argv[2] # 'rf', 'svm', 'lgrg', 'mlp'
dbname='gbmlgg_300x300_adaptive_ppi_gridrs1'
suffix = cnn + '_lr_rop_epoch_50_data_aug'
classes = ['0', '1', '2', '3', '4', '5']

# file format
# patchname, gt, <patch_level prediction histogram (6bin)>
src = 'features/'+dbname+'_macroArch_'+suffix+'_trainset_features.txt'
f = open(src)
train_content = f.readlines()
f.close()

src = 'features/'+dbname+'_macroArch_'+suffix+'_valset_features.txt'
f = open(src)
val_content = f.readlines()
f.close()

test_patchname_list = []
for line in val_content:
    parts = line.strip().split(',')
    test_patchname_list.append(parts[0])
#    gt = int(parts[1])
#    feat = [float(x) for x in parts[2:]]
#    X_test.append(feat)  
#    y_test.append(gt)

outfile = 'features/'+dbname+'_macroArch_'+suffix+'_trainval_features_reduced.npz'
npzdata = np.load(outfile)
X_train_wo_ps = npzdata['arr_0']
X_test = npzdata['arr_1']
y_train_wo_ps = npzdata['arr_2']
y_test = npzdata['arr_3']

print('X_test={}'.format(X_test.shape))

comb_id_list = list(range(1,126))                
                
ps_substr = cnn+'_data_aug_lr_rop_epoch50_net'

#for clfname in ['rf', 'svm', 'lgrg', 'mlp']:

save_path = 'ps_results/'+cnn+'/'+dbname+'_macroArch_'+suffix+'_patchlvl_'+clfname+'_testset_predictions_after_patch_selection.csv'
if not os.path.isdir(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
    
fp = open(save_path , 'w')

                
for comb_id in comb_id_list:
    
    src = 'patch_selections/'+cnn+'/'+dbname+'_macroArch_'+ps_substr+'_trainset_patch_selection'+str(comb_id)+'.pkl'      
    if not os.path.isdir(os.path.dirname(src)):
        os.makedirs(os.path.dirname(src))
    
    with open(src, 'rb') as h:  # Python 3: open(..., 'rb')
      selected_patches_dict, thld_list = pickle.load(h)  

    patch_name_list = list(selected_patches_dict.keys())

    selected_patch_count = 0
    selected_patch_list = []
    for patchname in selected_patches_dict:
        if selected_patches_dict[patchname][2] >= 0:
            selected_patch_count += 1              
            selected_patch_list.append(patchname)

    train_patchname_list, X_train,y_train = [],[],[]
    for i in range(len(train_content)):
        line = train_content[i]
        parts = line.strip().split(',')
        patchname = parts[0]
        if patchname in selected_patch_list:
#            gt = int(parts[1])
#            feat = [float(x) for x in parts[2:]]
#            X_train.append(feat)  
#            y_train.append(gt)
            X_train.append(X_train_wo_ps[i])  
            y_train.append(y_train_wo_ps[i])
            train_patchname_list.append(patchname)
    
    print('combid={}, len(train_content)={}, len(X_train)=({},{}), len(y_train)={}'.format(comb_id, len(train_content), len(X_train), len(X_train[0]), len(y_train)))


    #####################################################################        
    ## train and test with whole trainset
    if clfname == 'rf':
        
        clf = RandomForestClassifier()
        #plot_confusion_matrix(clf, X_test, y_test, normalize='true')  # doctest: +SKIP
        #plt.savefig('plots/'+dbname+'_macroArch_'+suffix+'_patchlvl_rf_confusion_matrix.png')
        #plt.show()  # doctest: +SKIP
        
    elif clfname == 'svm':
        
        #####################################################################        
        #print('grid searching optimal SVM params...')

        rand_idx = random.sample(range(len(X_train)),1000)
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

    elif clfname == 'lgrg':
        clf = LogisticRegression(max_iter=1500)
    elif clfname == 'mlp':
        clf = MLPClassifier(random_state=1, max_iter=500)            
    #####################################################################        

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    if not os.path.isdir('ps_output/'+cnn):
        os.makedirs('ps_output/'+cnn)

    np.savetxt('ps_output/'+cnn+'/'+dbname+'_macroArch_'+suffix+'_patchlvl_'+clfname+'_testset_predictions_after_patch_selection'+str(comb_id)+'.csv', [p for p in zip(test_patchname_list, y_test, y_pred, y_score)], delimiter=',', fmt='%s')

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
    
    print('comd_id:{} {} results: '.format(comb_id, clfname.upper()))
    print('valset: acc: {:.4f}, mAP(macro): {:.4f}, mAP(micro): {:.4f}, mAP(weighted): {:.4f}, mAP(samples): {:.4f}'.format(acc, mAP_macro, mAP_micro, mAP_wgt, mAP_smpls))
    print('confusion matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print_cm(cm, classes)
    print('confusion matrix (%):')
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print_cm(cm, classes)

    fp.write('comd_id:{} {} results: '.format(comb_id, clfname.upper()))
    fp.write('valset: acc: {:.4f}, mAP(macro): {:.4f}, mAP(micro): {:.4f}, mAP(weighted): {:.4f}, mAP(samples): {:.4f}'.format(acc, mAP_macro, mAP_micro, mAP_wgt, mAP_smpls))
    fp.write('confusion matrix:\n')
    fprint_cm(cm, classes, fp)
    fp.write('confusion matrix (%):\n')
    fprint_cm(cm, classes, fp)

    dump(clf, 'ps_models/'+dbname+'_macroArch_'+suffix+'_patchlvl_'+clfname+'_after_patch_selection'+str(comb_id)+'.model')


fp.close()



