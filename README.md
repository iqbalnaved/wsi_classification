# wsi_classfication
Patch based high resolution image classification

WSI classification project version 2.0
======================================

Changelog (08/14/2021):
* crop 300x300 size patches from WSIs
* random select 100 non blank patches from each images
* incremental radius for neighborhood search
Changelog (08/18/2021):
* grid wise random sampling. 5x5 grid size. 
* 400 ppi
* Set RandomCrop in transform preprocess stack (pytorch)
* Add mAP with all accracy prediction

Script run sequence:
====================

Preprocess:
-----------
detect_mostly_blank_imgs.py: generates histogram of ''total area of connected regions'', threshold is determined by manually examined the bins (see report)
create_patch_trainval.py: extract patches from wsi 

gridwise_random_select.py: randomly select x ppi that is divisible by 100 and greater than the value provided by Cochran's sample size formula, and save the selection list. If total_conn_comp < thresh, discard select another randomly, 
                  iteratively perform this until x-ppi is achieved.
create_trainvalsets.py: create the train /test directories, based on the random selection lists.
                  
CNN training:
------------

finetune_resnet50_cancer_data_aug.py: perform 100ppi, imagenet pretrained
finetune_inceptionv3_cancer_data_aug.py: perform 100ppi, imagenet pretrained
finetune_vgg16_cancer_data_aug.py: perform 100ppi, imagenet pretrained
finetune_vgg19bn_cancer_data_aug.py: perform 100ppi, imagenet pretrained

plot_trainval_log.py: plot from trainval log


CNN feature + Softmax --> class probabilities :
---------------------------------------------------------------------------

get_PatchLvl_Pred_resnet50.py: get softmax outputs
get_PatchLvl_Pred_inceptionv3.py: get softmax outputs
get_PatchLvl_Pred_vgg16.py: get softmax outputs
get_PatchLvl_Pred_vgg19bn.py: get softmax outputs

get_PatchLvl_Pred_Acc_mAP.py: get accuracy and mAP from softmax outputs (TODO: make generic)
get_ImgLevelVote_from_smx.py: majority vote of softmax winning class.

CNN feature + SVM/RF/LR/MLP --> Majority vote (without trainset patch selection):
-----------------------------------------------------------------------

extract_features_inceptionv3.py: extract features using inceptionV3
extract_features_resnet50.py
extract_features_vgg16.py
extract_features_vgg19bn.py
trainval_patchlvl_classifer.py: Apply PCA and train SVM,RF,LR,MLP and predict using cnn features
get_ImgLevelVote_from_classifier: get majority vote on valset


Trainset patch selection (and finding optimal patch selection thresholds)
-------------------------------------------------------------------------

get_rs_neighbor_info.py: get patch neighbors for all neighboring blocks.
perform_patch_selection_label_update_from_smx.py: perfrom trainset patch selection and label update from softmax classifier


CNN feature (same type used for PS) + classifier (all types) --> Majority vote (with trainset patch selection):
-----------------------------------------------------------------------

train_patchlvl_classifier_after_patch_selection_from_smx.py: train RF and predict using cnn (pca) features from selected patches
get_max_acc_comb_ids.py <..> pl-same: get list of combination ids comtaining the top-1 accuracy (output manually copied to next script)
get_ImgLevelVote_from_classifier_after_patch_selection: get majority vote on valset
get_max_acc_comb_ids.py <..> il-same: get list of combination ids comtaining the top-1 accuracy (output manually copied to next script)


Cross CNN patch selection --> Majority vote:
-------------------------------------------

train_patchlvl_classifier_after_patch_selection_from_smx_cross_cnn.py: train RF and predict using cnn features from selected patches
get_max_acc_comb_ids.py <..> pl-cross: get list of combination ids comtaining the top-1 accuracy (output manually copied to next script)
get_ImgLevelVote_from_classifier_after_ps_smx_cross_cnn.py : get majority vote on valset on the cross cnn
get_max_acc_comb_ids.py <..> il-cross: get list of combination ids comtaining the top-1 accuracy (output manually copied to next script)


Class probabilties -> region based softmax features -> Image-level SVM/RF/LR/MLP (after trainset patch selection):
-------------------------------------------------------------------------------------------------------------------

create_trainset_imglvl_dist_after_patch_selection.py:  create 1/4/9 distributions (train) after patch selection
create_valset_imglvl_dist_from_smx.py:  create 1/4/9 distributions (test)
trainval_ImgLevel_classifier_using_dist_from_smx_on_valset_after_patch_selection.py:  predict on test set, save accuracy for threshold adjustment
get_max_acc_comb_ids.py <..> x-rgn: get the approach with best accuracy

ensemble_thresh_comb.py : take average of probability predictions from top performing models (same model, different threshold combinations)

Analysis
--------

image_size_histogram: plots 2 figures, one containing frequency distribution of image area in 10 bins. Another plots frequency distribution of 
correctly classified labels (at image level) within those 10 bins in the first plot. 
plvl_rspatch_plot.py : plots points for discriminative patches correctly or incorrectly classified. plot title indicates image level classification
was correct or not. Non discriminative patches are also indicated. 
rspatch_plot.py: plots top-left points all randomly selected patches of a few WSIs. 

