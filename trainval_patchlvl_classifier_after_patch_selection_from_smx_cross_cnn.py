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


dbname='gbmlgg_300x300_adaptive_ppi_gridrs1'
classes = ['0', '1', '2', '3', '4', '5']

#cnn1 softmax predictions determines discriminative patch selection
cnn1 = sys.argv[1] #'resnet50' #'vgg19bn'  #'vgg16' #       'inceptionv3' # 
#cnn2 features are used for classification after patch selection
cnn2 = sys.argv[2] #'inceptionv3' #'vgg16' #resnet50'#'vgg19bn' #'resnet50' #'inceptionv3' #
suffix = cnn2 + '_lr_rop_epoch_50_data_aug'
clfname = sys.argv[3]
start_comb_id = int(sys.argv[4])
# cnn1 comb_id_list calculated using get_max_acc_comb_ids.py
#comb_id_list = [47, 108, 109, 22, 1, 98, 89, 28, 63, 87, 113, 55, 123, 125, 88, 82, 29, 33, 104, 85, 71, 114, 39, 100, 12, 96, 23] # inceptionv3
#comb_id_list = [113, 88, 111, 64, 112, 62]   # resnet50
#comb_id_list = [102, 9, 105, 1, 29, 4, 108, 107] # vgg16        
comb_id_list = list(range(start_comb_id, 126))

# file format
# patchname, gt, <patch_level prediction histogram (6bin)>
src = 'features/'+dbname+'_macroArch_'+suffix+'_trainset_features.txt' #cnn2
f = open(src)
train_content = f.readlines()
f.close()

train_patchname_list = [] #, X_test,y_test = [],[],[]
for line in train_content:
    parts = line.strip().split(',')
    train_patchname_list.append(parts[0])
del train_content # free memory
    
src = 'features/'+dbname+'_macroArch_'+suffix+'_valset_features.txt' # cnn2
f = open(src)
val_content = f.readlines()
f.close()

test_patchname_list = [] #, X_test,y_test = [],[],[]
for line in val_content:
    parts = line.strip().split(',')
    test_patchname_list.append(parts[0])
#    gt = int(parts[1])
#    feat = [float(x) for x in parts[2:]]
#    X_test.append(feat)  
#    y_test.append(gt)
del val_content # free memory

outfile = 'features/'+dbname+'_macroArch_'+suffix+'_trainval_features_reduced.npz'
npzdata = np.load(outfile)
X_train_wo_ps = npzdata['arr_0']
X_test = npzdata['arr_1']
y_train_wo_ps = npzdata['arr_2']
y_test = npzdata['arr_3']

del npzdata # free memory

save_path = 'ps_results/'+cnn1 +'_'+cnn2+'/'+dbname+'_macroArch_'+suffix+'_patchlvl_'+clfname+'_testset_predictions_after_patch_selection.csv'
if not os.path.isdir(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
    
fp = open(save_path , 'a') #'w')


ps_substr = cnn1+'_data_aug_lr_rop_epoch50_net'
                
for comb_id in comb_id_list: 
    
    src = 'patch_selections/'+cnn1+'/'+dbname+'_macroArch_'+ps_substr+'_trainset_patch_selection'+str(comb_id)+'.pkl'      
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

    X_train,y_train = [],[]
    for i in range(len(train_patchname_list)):
        patchname = train_patchname_list[i]
        if patchname in selected_patch_list:
#            gt = int(parts[1])
#            feat = [float(x) for x in parts[2:]]
#            X_train.append(feat)  
#            y_train.append(gt)
            X_train.append(X_train_wo_ps[i])  # using cnn features 
            y_train.append(y_train_wo_ps[i])

    
#    print('len(X_train)={}, len(y_train)={}'.format(len(X_train), len(y_train)))


#####################################################################        
## train and test with whole trainset
    if clfname == 'rf':
        clf = RandomForestClassifier()
    elif clfname == 'svm':
        print('grid searching optimal SVM params...')
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
        clf = LogisticRegression(max_iter=1000) #500)
    elif clfname == 'mlp':
        clf = MLPClassifier(random_state=1, max_iter=1000) #500)            

#####################################################################        

    clf.fit(X_train, y_train)
    del X_train # free mem
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    if not os.path.isdir('ps_output/'+cnn1+'_'+cnn2):
        os.makedirs('ps_output/'+cnn1+'_'+cnn2)
    np.savetxt('ps_output/'+cnn1+'_'+cnn2+'/'+dbname+'_macroArch_'+suffix+'_patchlvl_'+clfname+'_testset_predictions_after_patch_selection'+str(comb_id)+'.csv', [p for p in zip(test_patchname_list, y_test, y_pred, y_score)], delimiter=',', fmt='%s')

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
    cm1 = confusion_matrix(y_test, y_pred)
    print_cm(cm1, classes)
    print('confusion matrix (%):')
    cm2 = confusion_matrix(y_test, y_pred, normalize='true')
    print_cm(cm2, classes)

    fp.write('comd_id:{} {} results: '.format(comb_id, clfname.upper()))
    fp.write('Validation set accuracy: {:.4f}\n'.format(acc))
    fp.write('confusion matrix:\n')
    fprint_cm(cm1, classes, fp)
    fp.write('confusion matrix (%):\n')
    fprint_cm(cm2, classes, fp)

    #plot_confusion_matrix(clf, X_test, y_test, normalize='true')  # doctest: +SKIP
    #plt.savefig('plots/'+dbname+'_macroArch_'+suffix+'_patchlvl_rf_confusion_matrix.png')
    #plt.show()  # doctest: +SKIP


    if not os.path.isdir('ps_models/'+cnn1+'_'+cnn2):
        os.makedirs('ps_models/'+cnn1+'_'+cnn2)

    from joblib import dump
    dump(clf, 'ps_models/'+cnn1+'_'+cnn2+'/'+dbname+'_macroArch_'+suffix+'_patchlvl_'+clfname+'_after_patch_selection'+str(comb_id)+'.model')


fp.close()



