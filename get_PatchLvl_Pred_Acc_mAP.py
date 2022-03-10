import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sys

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
        
classes = ['0', '1','2','3', '4', '5']

dbname='gbmlgg_300x300_100ppi_rs1'
cnn = sys.argv[1]


print(cnn)
suffix = cnn + '_data_aug_lr_rop_epoch50'

src = 'features/'+dbname+'_macroArch_'+suffix+'_net_trainset_smx_predictions.txt'
f = open(src)
content = f.readlines()
f.close()

# patchname, ground_truth, predicted label, softmax
y_true, y_pred, y_score = [],[],[]
for line in content:
    parts = line.strip().split(',')
    patchname = parts[0]
    gt = int(parts[1])
    pred = int(parts[2])
    softmax = [float(x) for x in parts[3:]]
    y_true.append(gt)
    y_pred.append(pred)
    y_score.append(softmax)

    
yt_mtx = np.zeros((len(y_true), len(classes))) # size=(n_samples, n_classes)
ys_mtx = np.zeros((len(y_score), len(classes))) # size=(n_samples, n_classes)

for i in range(yt_mtx.shape[0]):
    yt_mtx[i, y_true[i]] = 1
    ys_mtx[i] = y_score[i]

acc = accuracy_score(y_true, y_pred)
ap_list_macro = average_precision_score(yt_mtx, ys_mtx, 'macro')
ap_list_micro = average_precision_score(yt_mtx, ys_mtx, 'micro')
ap_list_wgt = average_precision_score(yt_mtx, ys_mtx, 'weighted')
ap_list_smpls = average_precision_score(yt_mtx, ys_mtx, 'samples')
mAP_macro = np.mean(ap_list_macro)
mAP_micro = np.mean(ap_list_micro)
mAP_wgt = np.mean(ap_list_wgt)
mAP_smpls = np.mean(ap_list_smpls)

print('trainset: acc: {:.4f}, mAP(macro): {:.4f}, mAP(micro): {:.4f}, mAP(weighted): {:.4f}, mAP(samples): {:.4f}'.format(acc, mAP_macro, mAP_micro, mAP_wgt, mAP_smpls))
print('confusion matrix: ')
cm = confusion_matrix(y_test, y_pred)
print_cm(cm, classes)
src = 'features/'+dbname+'_macroArch_'+suffix+'_net_valset_smx_predictions.txt'
f = open(src)
content = f.readlines()
f.close()

# patchname, ground_truth, predicted label, softmax
y_true, y_pred, y_score = [],[],[]
for line in content:
    parts = line.strip().split(',')
    patchname = parts[0]
    gt = int(parts[1])
    pred = int(parts[2])
    softmax = [float(x) for x in parts[3:]]
    y_true.append(gt)
    y_pred.append(pred)
    y_score.append(softmax)

    
yt_mtx = np.zeros((len(y_true), len(classes))) # size=(n_samples, n_classes)
ys_mtx = np.zeros((len(y_score), len(classes))) # size=(n_samples, n_classes)

for i in range(yt_mtx.shape[0]):
    yt_mtx[i, y_true[i]] = 1
    ys_mtx[i] = y_score[i]

acc = accuracy_score(y_true, y_pred)
ap_list_macro = average_precision_score(yt_mtx, ys_mtx, 'macro')
ap_list_micro = average_precision_score(yt_mtx, ys_mtx, 'micro')
ap_list_wgt = average_precision_score(yt_mtx, ys_mtx, 'weighted')
ap_list_smpls = average_precision_score(yt_mtx, ys_mtx, 'samples')
mAP_macro = np.mean(ap_list_macro)
mAP_micro = np.mean(ap_list_micro)
mAP_wgt = np.mean(ap_list_wgt)
mAP_smpls = np.mean(ap_list_smpls)

print('valset: acc: {:.4f}, mAP(macro): {:.4f}, mAP(micro): {:.4f}, mAP(weighted): {:.4f}, mAP(samples): {:.4f}'.format(acc, mAP_macro, mAP_micro, mAP_wgt, mAP_smpls))
print('confusion matrix: ')
cm = confusion_matrix(y_test, y_pred)
print_cm(cm, classes)

