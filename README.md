# Efficient Classification of Histopathology Images Using Highly Imbalanced Data

**Mohammad Iqbal Nouyed**, M.-A. Hartley, G. Doretto, D. A. Adjeroh

*Pattern Recognition* (Springer LNCS), pp. 160–175, Dec. 2024. [doi:10.1007/978-3-031-78166-7_11](https://doi.org/10.1007/978-3-031-78166-7_11)

---

## Overview

Gigapixel whole-slide images (WSIs) present two compounding challenges for deep learning classifiers: extreme resolution and severe class imbalance. This repository contains the full patch-based classification pipeline from our study, which addresses both through discriminative patch selection, information-theoretic cluster-based sampling, and ensemble decision fusion.

**Key contributions:**
- Gridwise random patch sampling with Cochran's formula-based sample size guarantees
- Softmax-driven discriminative patch selection with neighborhood-aware label update
- Cross-CNN patch selection for model-agnostic discriminative region identification
- Region-based softmax feature distributions (1/4/9 spatial regions) for image-level classification
- Ensemble thresholding across top-performing model/threshold combinations

---

## Pipeline

```
1. Preprocessing
   WSIs → patch extraction → blank detection → gridwise random sampling

2. CNN Training (patch level)
   Sampled patches → fine-tuned CNN (ResNet50, InceptionV3, VGG16, VGG19-BN)

3. Patch-Level Prediction
   CNN softmax outputs → patch-level accuracy & mAP

4. Patch Selection (discriminative)
   Softmax outputs → neighbor-aware label update → selected discriminative patches

5. Image-Level Classification
   CNN features / softmax distributions → SVM/RF/LR/MLP → majority vote → ensemble
```

---

## Repository Structure

```
wsi_classfication/
│
├── Preprocessing
│   ├── extract_image_patches.py              # Extract patches from WSIs (.svs)
│   ├── extract_svs_info.py                   # Get WSI metadata
│   ├── detect_mostly_blank_imgs.py           # Blank patch detection via connected components
│   ├── create_blank_list.py                  # Build blank image list
│   ├── move_blanks.py                        # Move blank patches out of training set
│   ├── preprocess.py                         # General preprocessing utilities
│   ├── resize_images.py                      # Resize patches
│   ├── count_patches.py                      # Count patches per WSI
│   ├── get_patch_sizes.py                    # Patch size statistics
│   └── image_size_histogram.py              # WSI area frequency distribution
│
├── Patch Sampling
│   ├── gridwise_random_select.py             # Gridwise random sampling (Cochran's formula)
│   ├── random_select.py                      # Simple random patch selection
│   ├── create_patch_trainval.py              # Extract patches → train/val split
│   ├── create_patch_trainval_bot2top.py      # Bottom-to-top patch extraction variant
│   └── create_trainvalsets.py               # Build train/val directory structure
│
├── CNN Training
│   ├── finetune_resnet50_cancer_data_aug.py         # ResNet50 fine-tuning
│   ├── finetune_resnet50_cancer_data_aug_adam.py    # ResNet50 (Adam)
│   ├── finetune_resnet50_cancer_data_aug_adamW.py   # ResNet50 (AdamW)
│   ├── finetune_resnet34_cancer_data_aug.py         # ResNet34
│   ├── finetune_resnet18_cancer_data_aug.py         # ResNet18
│   ├── finetune_inceptionv3_cancer_data_aug.py      # InceptionV3
│   ├── finetune_vgg11bn_cancer_data_aug.py          # VGG11-BN
│   ├── finetune_vgg13bn_cancer_data_aug.py          # VGG13-BN
│   ├── finetune_vgg16_cancer_data_aug.py            # VGG16
│   └── finetune_vgg19bn_cancer_data_aug.py          # VGG19-BN
│
├── Patch-Level Prediction
│   ├── get_PatchLvl_Pred_resnet50.py         # ResNet50 softmax outputs
│   ├── get_PatchLvl_Pred_inceptionv3.py      # InceptionV3 softmax outputs
│   ├── get_PatchLvl_Pred_vgg16.py            # VGG16 softmax outputs
│   ├── get_PatchLvl_Pred_vgg19bn.py          # VGG19-BN softmax outputs
│   └── get_PatchLvl_Pred_Acc_mAP.py         # Patch-level accuracy and mAP
│
├── Feature Extraction
│   ├── extract_features_resnet50.py
│   ├── extract_features_inceptionv3.py
│   ├── extract_features_vgg16.py
│   └── extract_features_vgg19bn.py
│
├── Discriminative Patch Selection
│   ├── get_neighbor_info.py                  # Patch neighborhood structure
│   ├── get_rs_neighbor_info.py               # Neighborhood info for random-selected patches
│   └── perform_patch_selection_label_update_from_smx.py  # Softmax-driven PS + label update
│
├── Image-Level Classification
│   ├── trainval_patchlvl_classifier.py                          # PCA + SVM/RF/LR/MLP (no PS)
│   ├── trainval_patchlvl_classifier_after_patch_selection_from_smx.py        # With PS (same CNN)
│   ├── trainval_patchlvl_classifier_after_patch_selection_from_smx_cross_cnn.py  # Cross-CNN PS
│   ├── get_ImgLevelVote_from_classifier.py                      # Majority vote (no PS)
│   ├── get_ImgLevelVote_from_classifier_after_patch_selection.py             # Majority vote (PS)
│   ├── get_ImgLevelVote_from_classifier_after_ps_smx_cross_cnn.py           # Cross-CNN vote
│   ├── create_trainset_imglvl_dist_after_patch_selection.py     # 1/4/9 region distributions (train)
│   ├── create_valset_imglvl_dist_from_smx.py                   # 1/4/9 region distributions (val)
│   └── trainval_ImgLevel_classifier_using_dist_frm_smx_on_valset_after_patch_selection.py
│
├── Ensemble & Optimization
│   ├── get_max_acc_comb_ids.py               # Find top accuracy combination IDs
│   ├── ensemble_thresh_comb.py               # Ensemble by averaging top model probabilities
│   └── ensemble_thresh_comb2.py
│
├── Visualization & Analysis
│   ├── plot_trainval_log.py                  # Training/validation loss and accuracy curves
│   ├── plvl_rspatch_plot.py                  # Discriminative patch visualization per WSI
│   ├── rspatch_plot.py                       # Random patch spatial distribution plot
│   ├── psplot.py                             # Patch selection visualization
│   ├── sample_size_plot.py                   # Sample size analysis plot
│   └── image_size_histogram.py              # Image area frequency distribution
│
└── README.md
```

---

## Setup

```bash
git clone https://github.com/iqbalnaved/wsi_classfication.git
cd wsi_classfication
pip install torch torchvision openslide-python scikit-learn numpy pandas matplotlib pillow
```

---

## Usage

**Step 1 — Extract patches from WSIs**
```bash
python extract_image_patches.py
python detect_mostly_blank_imgs.py   # determine blank threshold from histogram
python create_blank_list.py
```

**Step 2 — Gridwise random sampling**
```bash
python gridwise_random_select.py     # generates selection lists per WSI
python create_trainvalsets.py        # builds train/val directory structure
```

**Step 3 — Fine-tune CNNs**
```bash
python finetune_resnet50_cancer_data_aug.py
python finetune_inceptionv3_cancer_data_aug.py
python finetune_vgg16_cancer_data_aug.py
python finetune_vgg19bn_cancer_data_aug.py
```

**Step 4 — Patch-level prediction**
```bash
python get_PatchLvl_Pred_resnet50.py
python get_PatchLvl_Pred_Acc_mAP.py
```

**Step 5 — Discriminative patch selection**
```bash
python get_rs_neighbor_info.py
python perform_patch_selection_label_update_from_smx.py
```

**Step 6 — Image-level classification**
```bash
# With patch selection (same CNN)
python trainval_patchlvl_classifier_after_patch_selection_from_smx.py
python get_max_acc_comb_ids.py       # identify top combinations
python get_ImgLevelVote_from_classifier_after_patch_selection.py

# Region-based softmax features
python create_trainset_imglvl_dist_after_patch_selection.py
python create_valset_imglvl_dist_from_smx.py
python trainval_ImgLevel_classifier_using_dist_frm_smx_on_valset_after_patch_selection.py
```

**Step 7 — Ensemble**
```bash
python ensemble_thresh_comb.py
```

---

## Citation

> Nouyed, M. I., Hartley, M.-A., Doretto, G., & Adjeroh, D. A. (2024). Efficient Classification of Histopathology Images Using Highly Imbalanced Data. *Pattern Recognition*, pp. 160–175. https://doi.org/10.1007/978-3-031-78166-7_11

---

## Related Work

- Nouyed et al. "A Framework for Evaluating Model Trustworthiness in Classification of Very High Resolution Histopathology Images." *IEEE BIBM*, 2024. [doi:10.1109/bibm62325.2024.10822778](https://doi.org/10.1109/bibm62325.2024.10822778)
- Nouyed et al. "Efficient Classification of Very High Resolution Histopathological Images." *IEEE BIBM*, 2022. [doi:10.1109/bibm55620.2022.9994942](https://doi.org/10.1109/bibm55620.2022.9994942)

---

## License

MIT License.
