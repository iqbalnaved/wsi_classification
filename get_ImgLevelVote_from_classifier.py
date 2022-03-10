# Majority voting without patch selection
# workflow
# patches -> cnn -> features -> svm -> patch lvl pred -> majority vote -> img lvl pred
import os
import glob
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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
        
dbname = 'gbmlgg_300x300_adaptive_ppi_gridrs1'
cnn = sys.argv[1]
suffix = cnn + '_lr_rop_epoch_50_data_aug'
classes = ['0', '1', '2', '3', '4', '5']
classifiers = ['rf'] # ['svm', 'rf', 'lgreg', 'mlp']

save_path = 'results/'+dbname+'_macroArch_'+suffix+'_net_valset_patchlvl_classifier_pred_majority_vote_accuracy.txt'

h = open(save_path, 'w')

for clf in classifiers:
    # file format: 
    # patchname, ground_truth, predicted label,softmax
#    src = 'output/Rail_300x300_wloc_macroArch_LeHou_Large_60ppi_lr_0.001_epoch_250_patchlvl_'+clf+'_testset_predictions.csv'
    src = 'output/'+dbname+'_macroArch_'+suffix+'_patchlvl_'+clf+'_testset_predictions.csv'
    f = open(src)
    content = f.readlines()
    f.close()

    print('Total number of valset patches: {}'.format(len(content)))

    pred_dict = {}
    gt_dict = {}
    for line in content:
        parts = line.strip().split(',')
        patchname = parts[0]
        gt = int(parts[1])
        pred = int(parts[2])
        imgname = '_'.join(patchname.split('_')[:-2])
        if imgname not in pred_dict:
            pred_dict[imgname] = []     
        pred_dict[imgname].append(pred)
        gt_dict[imgname] = int(gt) # no need for list, they're all same for each img

    #print('Total number of valset images: {}'.format(len(pred_dict)))

    img_level_pred_dict = {}    
    for imgname in pred_dict:
        mode = max(set(pred_dict[imgname]), key=pred_dict[imgname].count)    
        img_level_pred_dict[imgname] = mode


    dst = 'output/'+dbname+'_macroArch_'+suffix+'_net_valset_patchlvl_'+clf+'_pred_majority_vote.txt'
    f = open(dst, 'w')
    for imgname in pred_dict:
        f.write('{},{}\n'.format(imgname, pred_dict[imgname]))
    f.close()

    y_true, y_pred = [], []
    for filename in img_level_pred_dict:
        y_true.append(img_level_pred_dict[filename])
        y_pred.append(gt_dict[filename])

    y_true = [str(x) for x in y_true]
    y_pred = [str(x) for x in y_pred]    

    print('{} results:'.format(clf.upper()))
    acc = accuracy_score(y_true, y_pred)
    print('Validation set accuracy: {:.4f}'.format(acc))

    print('validation set confusion matrix: ')    
    cm_train = confusion_matrix(y_true, y_pred, classes)
    print_cm(cm_train, classes)
    print('validation set confusion matrix in percentage: ')    
    cm_train_n = confusion_matrix(y_true, y_pred, classes, normalize='true')
    print_cm(cm_train_n, classes)
    print()
    
    
    h.write('{} results:\n'.format(clf.upper()))
    h.write('Validation set accuracy: {:.4f}\n'.format(acc))
    h.write('validation set confusion matrix: \n')    
    fprint_cm(cm_train, classes, h)
    h.write('validation set confusion matrix in percentage: \n')    
    fprint_cm(cm_train_n, classes, h)
    h.write('\n')
    
h.close()    
        
