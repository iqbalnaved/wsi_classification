# Majority voting without patch selection
# workflow
# patches -> cnn -> features -> svm -> patch lvl pred -> majority vote -> img lvl pred
import os
import glob
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
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

def fprint_cm(cm, labels, fp, hide_zeroes=False, hide_diagonal=False, hide_threshold=None ):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    fp.write("    " + empty_cell)
    for label in labels:
        fp.write("%{0}s".format(columnwidth) % label)
    fp.write('\n')
    # Print rows
    for i, label1 in enumerate(labels):
        fp.write("    %{0}s".format(columnwidth) % label1)
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
            fp.write(cell)
        fp.write('\n')
                
classes = ['0', '1', '2', '3', '4', '5']
cnn = sys.argv[1] #'vgg19bn_vgg16' # 'vgg19bn_inceptionv3' #'vgg19bn_resnet50' #'vgg16_vgg19bn' #'vgg16_resnet50' #'vgg16_inceptionv3' #'resnet50_vgg19bn' #'resnet50_vgg16' #'resnet50_inceptionv3' #'inceptionv3_vgg19bn' #'inceptionv3_vgg16' #'inceptionv3_resnet50' #
cnn2 = cnn.split('_')[1]
clfname = sys.argv[2]
dbname = 'gbmlgg_300x300_adaptive_ppi_gridrs1'
suffix = cnn2 + '_lr_rop_epoch_50_data_aug'
dst = 'ps_results/'+cnn+'/'+dbname+'_macroArch_'+suffix+'_net_valset_patchlvl_'+clfname+'_pred_after_ps_majority_vote.txt'
fp = open(dst, 'w')
#    for imgname in pred_dict:
#        f.write('{},{}\n'.format(imgname, pred_dict[imgname]))
#    f.close()                
#csv_list = glob.glob(os.path.join('ps_output/'+cnn, dbname + '_macroArch_'+suffix+'_patchlvl_'+clfname+'_testset_predictions_after_patch_selection*.csv')) #
comb_id_list = list(range(1,126)) 

for comb_id in comb_id_list:
#    print('csv: {}'.format(csv))
#    fp.write('csv: {}\n'.format(csv))

#    src = 'patch_selections/smx6/Rail_macroArch_LeHou_Large_60ppi_lr_0.001_epoch250_net_trainset_patch_selection'+str(comb_id)+'.pkl'      
#    
#    with open(src, 'rb') as h:  # Python 3: open(..., 'rb')
#      selected_patches_dict, thld_list = pickle.load(h)  
#    print('thresholds:{}'.format(str(thld_list)))
#    fp.write('thresholds:{}\n'.format(str(thld_list)))

    # file format: 
    # patchname, ground_truth, predicted label,softmax
    #src = '/home/adjeroh/extreme_scales/project_rail300x300/rail_300x300_LeHou_Larger_patchlvl_svm_testset_predictions.csv'
    #src = '/home/adjeroh/extreme_scales/output/Rail_macroArch_LeHou_Large_60ppi_lr_0.001_epoch250_net_valset_predictions.txt'
    src = 'ps_output/'+cnn+'/'+dbname+'_macroArch_'+suffix+'_patchlvl_'+clfname+'_testset_predictions_after_patch_selection'+str(comb_id)+'.csv'

    f = open(src)
    content = f.readlines()
    f.close()

    print('combId:{} Total number of valset patches: {}'.format(comb_id, len(content)))
    fp.write('combId:{} Total number of valset patches: {}\n'.format(comb_id, len(content)))

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

    y_true, y_pred = [], []
    for filename in img_level_pred_dict:
        y_true.append(img_level_pred_dict[filename])
        y_pred.append(gt_dict[filename])

    y_true = [str(x) for x in y_true]
    y_pred = [str(x) for x in y_pred]    

    acc = accuracy_score(y_true, y_pred)
    print('Validation set {} accuracy: {:.4f}'.format(clfname.upper(), acc))
    fp.write('Validation set {} accuracy: {:.4f}\n'.format(clfname.upper(), acc))

    print('validation set confusion matrix: ')    
    fp.write('validation set confusion matrix: \n')    
    cm_train = confusion_matrix(y_true, y_pred, classes)
    print_cm(cm_train, classes)
    fprint_cm(cm_train, classes, fp)
    print('validation set confusion matrix in percentage: ')    
    fp.write('validation set confusion matrix in percentage: \n')    
    cm_train = confusion_matrix(y_true, y_pred, classes, normalize='true')
    print_cm(cm_train, classes)
    fprint_cm(cm_train, classes, fp)    


fp.close()
