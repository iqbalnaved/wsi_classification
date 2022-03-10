# get the patch level predictions
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time 
import numpy as np
from torchvision.datasets.folder import default_loader
import os, sys
import glob
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F

rs_id = sys.argv[1]

dbname='gbmlgg_300x300_adaptive_ppi_gridrs'+rs_id
suffix = 'resnet50_data_aug'
#learning_rate=0.001
classes = [0, 1, 2, 3, 4, 5]
#num_epoch=15     
num_classes = len(classes)

test_transform = transforms.Compose(
    [#transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#2. Define a Convolutional Neural Network
#take 3-channel images (instead of 1-channel images as it was defined).

#import torch.nn as nn
#import torch.nn.functional as F

###################### PASTE MODEL CLASS HERE #####################

from torchvision import models

net = models.resnet50()
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, num_classes)


PATH = 'models/resnet50_imagenet_gbmlgg_300x300_adaptive_ppi_gridrs'+rs_id+'_net.pth'
#checkpoint = torch.load(PATH) # load saved checkpoint model
#net.load_state_dict(checkpoint['state_dict'])
net.load_state_dict(torch.load(PATH))
net.eval()      
net.to(device)

###################### PASTE MODEL CLASS HERE #####################

     
f = open('features/'+dbname+'_macroArch'+suffix+'_lr_rop_epoch50_net_trainset_smx_predictions.txt', 'w')

directory = "/largeDataVolume/LeHou_GBMLGG_300x300_adaptive_ppi_gridwise_random_"+rs_id+"/train"
dirlist = os.listdir(directory)
#y_train_pred_dict = {}
for dirname in dirlist:
    filelist = glob.glob(os.path.join(directory, dirname, '*.png'))
    for filename in filelist:
#        school_id = filename[0:6]
#        ids.append(school_id)
#        to_open = filename
        img = Image.open(filename)
        img = img.convert('RGB')
#        transform = transforms.Compose([
#            transforms.Resize(256),
#            transforms.CenterCrop(224),         
#            transforms.ToTensor(),
#            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        ])

        img_t = test_transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
#        net.eval()
        out = net(batch_t.to(device))
        out = F.softmax(out, dim=-1) # added softmax 

        _, index = torch.max(out, 1)

        patchname = os.path.basename(filename).split('.')[0]
#        y_train_pred_dict[patchname] = (int(dirname), index.item())
        print("{}, {}, {}, {}".format(patchname, dirname, str(index.item()), str(out.tolist()).strip('[]'))) # patchname, ground_truth, predicted label, softmax
        f.write("{}, {}, {}, {}\n".format(patchname, dirname, str(index.item()), str(out.tolist()).strip('[]')))
                
#        percentage = torch.nn.functional.softmax(out, dim=1)[0]
#        class0.append(percentage[0].tolist())
#        class1.append(percentage[1].tolist())
f.close()


###########################################################################
f = open('features/'+dbname+'_macroArch'+suffix+'_lr_rop_epoch50_net_valset_smx_predictions.txt', 'w')

directory = "/largeDataVolume/LeHou_GBMLGG_300x300_adaptive_ppi_gridwise_random_"+rs_id+"/test"
dirlist = os.listdir(directory)
#y_val_pred_dict = {}
for dirname in dirlist:
    filelist = glob.glob(os.path.join(directory, dirname, '*.png'))
    for filename in filelist:
#        school_id = filename[0:6]
#        ids.append(school_id)
#        to_open = filename
        img = Image.open(filename)
        img = img.convert('RGB')
#        transform = transforms.Compose([
#            transforms.Resize(256),
#            transforms.CenterCrop(224),         
#            transforms.ToTensor(),
#            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#        ])
        img_t = test_transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
#        net.eval()
        out = net(batch_t.to(device))
        out = F.softmax(out, dim=-1)
        
        _, index = torch.max(out, 1)
        
        patchname = os.path.basename(filename).split('.')[0]
#        y_val_pred_dict[patchname] = (int(dirname), index.item())
        print("{}, {}, {}, {}".format(patchname, dirname, str(index.item()), str(out.tolist()).strip('[]'))) # patchname, ground_truth, predicted label, softmax
        f.write("{}, {}, {}, {}\n".format(patchname, dirname, str(index.item()), str(out.tolist()).strip('[]')))
                            
#        percentage = torch.nn.functional.softmax(out, dim=1)[0]
#        class0.append(percentage[0].tolist())
#        class1.append(percentage[1].tolist())

f.close()


