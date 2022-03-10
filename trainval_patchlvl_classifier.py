# takes patch level deep features as input
# trains svm on trainset features (patch level)
# predicts on valset features (patch level)
import os, glob
import random
import numpy as np
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # doctest: +SKIP
from sklearn.metrics import plot_confusion_matrix

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

# create confusion matrices for patch level prediction 
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


cnn = sys.argv[1]
clfname = sys.argv[2] # options: 'all' 'svm' 'rf' 'lgrg' 'mlp'
dopca = sys.argv[3] # 'yes' 'no' (load)

#dbname='Rail_300x300_ppi'
dbname='gbmlgg_300x300_adaptive_ppi_gridrs1'
#suffix = 'inceptionv3_lr_rop_epoch50_data_aug'
suffix = cnn + '_lr_rop_epoch_50_data_aug'

classes = ['0', '1', '2', '3', '4', '5']


print('loading train and test datasets...') # for pca , one time only

# file format
# imgname, gt, <patch_level prediction histogram (6bin)>
# one distibution per image
#src = '/home/adjeroh/extreme_scales/output/Rail_macroArch_LeHou_Large_trainset_features.txt'
#src = '/home/adjeroh/extreme_scales/output/Rail_macroArch_LeHou_Large_60ppi_trainset_features.txt'
src = 'features/'+dbname+'_macroArch_'+suffix+'_trainset_features.txt'
f = open(src)
content = f.readlines()
f.close()

train_patchname_list, X_train,y_train = [],[],[]
for line in content:
    parts = line.strip().split(',')
    train_patchname_list.append(parts[0])
    gt = int(parts[1])
    feat = [float(x) for x in parts[2:]]
    X_train.append(feat)  
    y_train.append(gt)

del content # free memory
#src = '/home/adjeroh/extreme_scales/output/Rail_macroArch_LeHou_Large_60ppi_valset_features.txt'
src = 'features/'+dbname+'_macroArch_'+suffix+'_valset_features.txt'
f = open(src)
content = f.readlines()
f.close()

test_patchname_list, X_test,y_test = [],[],[]
for line in content:
    parts = line.strip().split(',')
    test_patchname_list.append(parts[0])
    gt = int(parts[1])
    feat = [float(x) for x in parts[2:]]
    X_test.append(feat)  
    y_test.append(gt)

del content # free memory
print('Before PCA: X_train.shape=({},{}), X_test.shape=({},{})'.format(len(X_train), len(X_train[9]), len(X_test), len(X_test[9])))



    #####################################################################
if dopca == 'yes':    
    print('Apply PCA')

    X = np.concatenate((np.array(X_train), np.array(X_test)), axis=0) #[X_train;X_test]
    print('X.shape=({},{})'.format(len(X), len(X[0])))

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_rescaled = scaler.fit_transform(X)
    del X # free memory
    #99% of variance
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 0.99)
    pca.fit(X_rescaled)
    X_reduced = pca.transform(X_rescaled)
    del X_rescaled # free memory
    X_train = X_reduced[:len(y_train)]
    X_test = X_reduced[len(y_train):]

    print('After PCA: X_train.shape=({},{}), X_test.shape=({},{})'.format(len(X_train), len(X_train[9]), len(X_test), len(X_test[9])))

    outfile = 'features/'+dbname+'_macroArch_'+suffix+'_trainval_features_reduced.npz'
    np.savez(outfile, X_train, X_test, y_train, y_test)
else:
    print('loading saved pca features..')
    pcafile = 'features/'+dbname+'_macroArch_'+suffix+'_trainval_features_reduced.npz'
    npzdata = np.load(pcafile)
    X_train = npzdata['arr_0']
    X_test = npzdata['arr_1']
    y_train = npzdata['arr_2']
    y_test = npzdata['arr_3']

    print('PCA feat. sz.: X_train.shape=({},{}), X_test.shape=({},{})'.format(len(X_train), len(X_train[9]), len(X_test), len(X_test[9])))

h = open('results/'+dbname+'_'+suffix+'_patchlvl_'+clfname+'_classifier_results.txt', 'w')    
#####################################################################        
if clfname == 'all' or clfname == 'svm': 
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

    #####################################################################        
    ## train and test with whole trainset
            
    clf = svm.SVC(C_val)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred2 = clf.predict(X_train)
    np.savetxt('output/'+dbname+'_macroArch_'+suffix+'_patchlvl_svm_testset_predictions.csv', [p for p in zip(test_patchname_list, y_test, y_pred)], delimiter=',', fmt='%s')
    np.savetxt('output/'+dbname+'_macroArch_'+suffix+'_patchlvl_svm_trainset_predictions.csv', [p for p in zip(train_patchname_list, y_train, y_pred2)], delimiter=',', fmt='%s')

    print('SVM results: ')
    acc = accuracy_score(y_test, y_pred)
    print('Validation set accuracy: {:.4f}'.format(acc))
    print('confusion matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print_cm(cm, classes)
    print('confusion matrix (%):')
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print_cm(cm, classes)

    h.write('SVM results: \n')
    h.write('Validation set accuracy: {:.4f}\n'.format(acc))
    h.write('confusion matrix:\n')
    fprint_cm(cm, classes, h)
    h.write('confusion matrix (%):\n')
    fprint_cm(cm, classes, h)

    #plot_confusion_matrix(clf, X_test, y_test, normalize='true')  # doctest: +SKIP
    #plt.savefig('plots/'+dbname+'_macroArch_'+suffix+'_patchlvl_svm_confusion_matrix.png')
    #plt.show()  # doctest: +SKIP


    from joblib import dump
    dump(clf, 'models/'+dbname+'_macroArch_'+suffix+'_patchlvl_svm.model')


#####################################################################  
if clfname == 'all' or clfname == 'rf': 

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred2 = clf.predict(X_train)
    
    y_score = clf.predict_proba(X_test)
    y_score2 = clf.predict_proba(X_train)
        
    np.savetxt('output/'+dbname+'_macroArch_'+suffix+'_patchlvl_rf_testset_predictions.csv', [p for p in zip(test_patchname_list, y_test, y_pred, y_score)], delimiter=',', fmt='%s')
    np.savetxt('output/'+dbname+'_macroArch_'+suffix+'_patchlvl_rf_trainset_predictions.csv', [p for p in zip(train_patchname_list, y_train, y_pred2, y_score2)], delimiter=',', fmt='%s')

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

    print('RF results: ')
    print('valset: acc: {:.4f}, mAP(macro): {:.4f}, mAP(micro): {:.4f}, mAP(weighted): {:.4f}, mAP(samples): {:.4f}'.format(acc, mAP_macro, mAP_micro, mAP_wgt, mAP_smpls))
    print('confusion matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print_cm(cm, classes)
    print('confusion matrix (%):')
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print_cm(cm, classes)

    h.write('RF results: \n')
    h.write('Validation set accuracy: {:.4f}\n'.format(acc))
    h.write('confusion matrix:\n')
    fprint_cm(cm, classes, h)
    h.write('confusion matrix (%):\n')
    fprint_cm(cm, classes, h)

    #plot_confusion_matrix(clf, X_test, y_test, normalize='true')  # doctest: +SKIP
    #plt.savefig('plots/'+dbname+'_macroArch_'+suffix+'_patchlvl_rf_confusion_matrix.png')
    #plt.show()  # doctest: +SKIP


    from joblib import dump
    dump(clf, 'models/'+dbname+'_macroArch_'+suffix+'_patchlvl_rf.model')

#####################################################################  
if clfname == 'all' or clfname == 'lgrg': 
    #clf = LogisticRegression(max_iter=500) # inceptionv3, 65ppi
    clf = LogisticRegression(max_iter=1500)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred2 = clf.predict(X_train)

    y_score = clf.predict_proba(X_test)
    y_score2 = clf.predict_proba(X_train)
            
    np.savetxt('output/'+dbname+'_macroArch_'+suffix+'_patchlvl_lgreg_testset_predictions.csv', [p for p in zip(test_patchname_list, y_test, y_pred, y_score)], delimiter=',', fmt='%s')
    np.savetxt('output/'+dbname+'_macroArch_'+suffix+'_patchlvl_lgreg_trainset_predictions.csv', [p for p in zip(train_patchname_list, y_train, y_pred2, y_score2)], delimiter=',', fmt='%s')

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

    print('Logistic Reg. results: ')
    print('valset: acc: {:.4f}, mAP(macro): {:.4f}, mAP(micro): {:.4f}, mAP(weighted): {:.4f}, mAP(samples): {:.4f}'.format(acc, mAP_macro, mAP_micro, mAP_wgt, mAP_smpls))
    print('confusion matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print_cm(cm, classes)
    print('confusion matrix (%):')
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print_cm(cm, classes)

    h.write('Logistic Reg. results: \n')
    h.write('Validation set accuracy: {:.4f}\n'.format(acc))
    h.write('confusion matrix:\n')
    fprint_cm(cm, classes, h)
    h.write('confusion matrix (%):\n')
    fprint_cm(cm, classes, h)

    #plot_confusion_matrix(clf, X_test, y_test, normalize='true')  # doctest: +SKIP
    #plt.savefig('plots/'+dbname+'_macroArch_'+suffix+'_patchlvl_lgreg_confusion_matrix.png')
    #plt.show()  # doctest: +SKIP


    from joblib import dump
    dump(clf, 'models/'+dbname+'_macroArch_'+suffix+'_patchlvl_lgreg.model')

#####################################################################  

if clfname == 'all' or clfname == 'mlp': 
    clf = MLPClassifier(random_state=1, max_iter=500)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred2 = clf.predict(X_train)
    
    y_score = clf.predict_proba(X_test)
    y_score2 = clf.predict_proba(X_train)
        
    np.savetxt('output/'+dbname+'_macroArch_'+suffix+'_patchlvl_mlp_testset_predictions.csv', [p for p in zip(test_patchname_list, y_test, y_pred, y_score)], delimiter=',', fmt='%s')
    np.savetxt('output/'+dbname+'_macroArch_'+suffix+'_patchlvl_mlp_trainset_predictions.csv', [p for p in zip(train_patchname_list, y_train, y_pred2, y_score2)], delimiter=',', fmt='%s')

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


    print('MLP results: ')
    print('valset: acc: {:.4f}, mAP(macro): {:.4f}, mAP(micro): {:.4f}, mAP(weighted): {:.4f}, mAP(samples): {:.4f}'.format(acc, mAP_macro, mAP_micro, mAP_wgt, mAP_smpls))
    print('Validation set accuracy: {:.4f}'.format(acc))
    print('confusion matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print_cm(cm, classes)
    print('confusion matrix (%):')
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print_cm(cm, classes)

    h.write('MLP results: \n')
    h.write('Validation set accuracy: {:.4f}\n'.format(acc))
    h.write('confusion matrix:\n')
    fprint_cm(cm, classes, h)
    h.write('confusion matrix (%):\n')
    fprint_cm(cm, classes, h)
    #plot_confusion_matrix(clf, X_test, y_test, normalize='true')  # doctest: +SKIP
    #plt.savefig('plots/'+dbname+'_macroArch_'+suffix+'_patchlvl_mlp_confusion_matrix.png')
    #plt.show()  # doctest: +SKIP


    from joblib import dump
    dump(clf, 'models/'+dbname+'_macroArch_'+suffix+'_patchlvl_mlp.model')

h.close()
