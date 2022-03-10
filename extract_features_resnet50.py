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
suffix = 'resnet50_lr_rop_epoch_50_data_aug'
#learning_rate=0.001
classes = [0, 1, 2, 3, 4, 5]
#num_epoch=500     
num_classes = len(classes)

test_transform = transforms.Compose(
    [#transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#2. Define a Convolutional Neural Network
#take 3-channel images (instead of 1-channel images as it was defined).

from torchvision import models

class Resnet50Extractor(nn.Module):
    def __init__(self, submodule, extracted_layer):
        super(Resnet50Extractor, self).__init__()
        self.submodule = submodule
        self.extracted_layer = extracted_layer

    def forward(self, x):
        if self.extracted_layer == 'maxpool':                       
            modules = list(self.submodule.children())[:4]
        elif self.extracted_layer == 'inner-layer-3':                     
            modules = list(self.submodule.children())[:6]
            third_module = list(self.submodule.children())[6]
            third_module_modules = list(third_module.children())[:3]    # take the first three inner modules
            third_module = nn.Sequential(*third_module_modules)
            modules.append(third_module)
        elif self.extracted_layer == 'layer-3':                     
            modules = list(self.submodule.children())[:7]
        else:                                               # after avg-pool
            modules = list(self.submodule.children())[:9]

        self.submodule = nn.Sequential(*modules)
        x = self.submodule(x)
        return x

model_ft = models.resnet50()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

# Rail_macroArch_LeHou_Large_lr_0.001_epoch_500_net.pth
#PATH = './'+dbname+'_macroArch'+suffix+'_lr_'+str(learning_rate)+'_epoch_'+str(num_epoch)+'_net.pth'
PATH = 'models/resnet50_imagenet_gbmlgg_300x300_adaptive_ppi_gridrs'+rs_id+'_net.pth'
#checkpoint = torch.load(PATH) # load saved checkpoint model
#net.load_state_dict(checkpoint['state_dict'])
model_ft.load_state_dict(torch.load(PATH))
model_ft.eval()      
#net.to(device)
extracted_layer = 'after-avg-pool'
extractor = Resnet50Extractor(model_ft, extracted_layer)

#input_tensor = torch.rand(1, 3, 224, 224)
#features = extractor(input_tensor)
#features = torch.squeeze(features,2)
#features = torch.squeeze(features,2)
##f = net(torch.rand(1, 3, 224, 224)) 
#print(features.shape) #utput: torch.Size([1, 2048])

#import sys
#sys.exit()
     
f = open('features/'+dbname+'_macroArch_'+suffix+'_trainset_features.txt', 'w')

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
#        out = net(batch_t.to(device))
        features = extractor(batch_t)
        features = torch.squeeze(features,2)
        out = torch.squeeze(features,2)

        _, index = torch.max(out, 1)

        patchname = os.path.basename(filename).split('.')[0]
#        y_train_pred_dict[patchname] = (int(dirname), index.item())
        print("{}, {}, {}".format(patchname, dirname, str(out.tolist()).strip('[]'))) # patchname, ground_truth, predicted label, softmax
        f.write("{}, {}, {}\n".format(patchname, dirname, str(out.tolist()).strip('[]')))
                
#        percentage = torch.nn.functional.softmax(out, dim=1)[0]
#        class0.append(percentage[0].tolist())
#        class1.append(percentage[1].tolist())
f.close()


###########################################################################
f = open('features/'+dbname+'_macroArch_'+suffix+'_valset_features.txt', 'w')
c=0
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
#        out = net(batch_t.to(device))
        features = extractor(batch_t)
        features = torch.squeeze(features,2)
        out = torch.squeeze(features,2)

        _, index = torch.max(out, 1)
        
        patchname = os.path.basename(filename).split('.')[0]
#        y_val_pred_dict[patchname] = (int(dirname), index.item())
        print("{}, {}, {}".format(patchname, dirname, str(out.tolist()).strip('[]'))) # patchname, ground_truth, predicted label, softmax
        f.write("{}, {}, {}\n".format(patchname, dirname, str(out.tolist()).strip('[]')))
                            
#        percentage = torch.nn.functional.softmax(out, dim=1)[0]
#        class0.append(percentage[0].tolist())
#        class1.append(percentage[1].tolist())

f.close()


