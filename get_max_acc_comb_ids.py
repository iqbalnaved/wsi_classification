from operator import itemgetter
import sys
####################
# usage:
# python3 <cnn>  <clf> pl-same
# python3 <cnn>  <clf> il-same
# python3 <cnn1>_<cnn2> <clf> pl-cross
# python3 <cnn1>_<cnn2> <clf> il-cross
# python3 <cnn1>_<cnn2> <clf> 1-rgn
# python3 <cnn1>_<cnn2> <clf> 4-rgn
# python3 <cnn1>_<cnn2> <clf> 9-rgn

cnn =  sys.argv[1] # for cross cnn pass like this e.g. resnet50_inceptionv3 ('underscore' is the delimiter)
clf = sys.argv[2]
rtype = sys.argv[3] # 'pl-same' 'il-same' 'pl-cross' 'il-cross' 'x-rgn'

dbname = 'gbmlgg_300x300_adaptive_ppi_gridrs1'

if rtype == 'pl-same':
    src = 'ps_results/'+cnn+'/'+dbname+'_macroArch_'+cnn+'_lr_rop_epoch_50_data_aug_patchlvl_'+clf+'_testset_predictions_after_patch_selection.csv'
elif rtype == 'il-same':
    src = 'ps_results/'+cnn+'/'+dbname+'_macroArch_'+cnn+'_lr_rop_epoch_50_data_aug_net_valset_patchlvl_'+clf+'_pred_after_ps_majority_vote.txt'
#################
# cros-cnn 
elif rtype == 'pl-cross':
    #patch level
    #cnn1 = 'inceptionv3' #'vgg16' #'vgg19bn' #'resnet50' 
    cnn2 = cnn.split('_')[1] #'resnet50' #'vgg19bn' #'vgg16' #'inceptionv3' #'inceptionv3' #'vgg16' #'resnet50'
    # clf = ''
    #cnn = cnn1 + '_' + cnn2 
    src = 'ps_results/'+cnn+'/'+dbname+'_macroArch_'+cnn2+'_lr_rop_epoch_50_data_aug_patchlvl_'+clf+'_testset_predictions_after_patch_selection.csv'
elif rtype == 'il-cross':
# img level
    cnn2 = cnn.split('_')[1] #'resnet50' #'vgg19bn' #'vgg16' #'inceptionv3' #'inceptionv3' #'vgg16' #'resnet50'
#src = 'ps_results/'+cnn+'/Rail_300x300_65ppi_macroArch_'+cnn+'_65ppi_lr_rop_epoch50_data_aug_net_valset_patchlvl_rf_pred_after_ps_majority_vote.txt'
    src = 'ps_results/'+cnn+'/'+dbname+'_macroArch_'+cnn2+'_lr_rop_epoch_50_data_aug_net_valset_patchlvl_'+clf+'_pred_after_ps_majority_vote.txt'
########################
elif rtype == '1-rgn':
    src = 'ps_results/'+cnn+'_'+clf+'/'+dbname+'_'+cnn+'_'+clf+'_valset_softmax_1_region_avg_dist_prediction_after_ps.txt'
elif rtype == '4-rgn':
    src = 'ps_results/'+cnn+'_'+clf+'/'+dbname+'_'+cnn+'_'+clf+'_valset_softmax_4_region_avg_dist_prediction_after_ps.txt'
elif rtype == '9-rgn':
    src = 'ps_results/'+cnn+'_'+clf+'/'+dbname+'_'+cnn+'_'+clf+'_valset_softmax_9_region_avg_dist_prediction_after_ps.txt'

########################

print(cnn)

f = open(src)
content = f.readlines()
f.close()

accuracy_dict ={}
for i in range(len(content)):
    line = content[i]
    if rtype == 'il-same':
        if 'accuracy:' in line: 
            acc = line.strip().split(':')[1].strip()
            comid = int(content[i-2].strip().split(':')[1].strip())
    #        comid= content[i-2].strip().split('selection')[1].split('.')[0]
            accuracy_dict[comid] = float(acc)
    elif rtype == 'pl-same':
        if "comd_id:" in line:
            parts = line.split()            
            acc = parts[5].strip(', ')
            comid = int(parts[0].split(':')[1])
            accuracy_dict[comid] = float(acc)
    elif rtype == 'pl-cross':
        if "comd_id:" in line:
            parts = line.split()            
            acc = parts[6].strip(', ')
            comid = int(parts[0].split(':')[1])
            accuracy_dict[comid] = float(acc)
#        acc = line.strip().split(':')[-1].strip()
#        comid =  int(line.strip().split(' ')[0].split(':')[1])
#        accuracy_dict[comid] = float(acc)
    elif rtype == 'il-cross':
#        acc = line.strip().split(':')[1].strip()
#        comid = int(content[i-1].strip().split(' ')[0].split(':')[1])
#        accuracy_dict[comid] = float(acc)
        if 'accuracy:' in line: 
            acc = line.strip().split(':')[1].strip()
            comid = int(content[i-1].split()[0].split(':')[1].strip())
    #        comid= content[i-2].strip().split('selection')[1].split('.')[0]
            accuracy_dict[comid] = float(acc)
    elif rtype == '1-rgn' or rtype == '4-rgn' or rtype == '9-rgn':
        if 'accuracy:' in line: 
            acc = line.split()[4].strip(', ') #line.split()[0].split(':')[1].strip()
            comid = int(content[i-1].strip().split(':')[1])        
            accuracy_dict[comid] = float(acc)

        
        
res = list(sorted(accuracy_dict.items(), key=itemgetter(1), reverse=True))

print(res)

maxval = max(accuracy_dict.values())
print('max value: {}'.format(maxval))

max_combid_list = []
combid_list = []
for tupl in res:
    if tupl[1] == maxval:
        max_combid_list.append(tupl)
        combid_list.append(int(tupl[0]))
        
print(max_combid_list)        

print(combid_list)



