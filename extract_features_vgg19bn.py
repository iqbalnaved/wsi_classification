# get the patch level predictions
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time 
import numpy as np
from torchvision.datasets.folder import default_loader
import os
import glob
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

dbname='gbmlgg_300x300_adaptive_ppi_gridrs1'
suffix = '_vgg19bn_lr_rop_epoch_50_data_aug'
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
from torchvision.models.vgg import VGG
class MyVgg(VGG):

    def __init__(self):
        super().__init__(models.vgg19_bn())
        net = models.vgg19_bn()
        self.features=net.features
    def forward(self, x):
        # here, implement the forward function, keep the feature maps you like
        # and return them
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)        
        x = self.classifier[3](x)
        return x

net = MyVgg()
#net = models.vgg19_bn()
net.classifier[6] = nn.Linear(4096,num_classes)



# Rail_macroArch_LeHou_Large_lr_0.001_epoch_500_net.pth
#PATH = './'+dbname+'_macroArch'+suffix+'_lr_'+str(learning_rate)+'_epoch_'+str(num_epoch)+'_net.pth'
PATH = 'models/vgg19bn_imagenet_gbmlgg_300x300_adaptive_ppi_gridrs1_net.pth'
#checkpoint = torch.load(PATH) # load saved checkpoint model
#net.load_state_dict(checkpoint['state_dict'])
net.load_state_dict(torch.load(PATH))
net.eval()      
net.to(device)

#f = net(torch.rand(1, 3, 224, 224)) 
#print(f.shape) #utput: torch.Size([1, 4096])

#import sys
#sys.exit()
     
f = open('features/'+dbname+'_macroArch'+suffix+'_trainset_features.txt', 'w')

directory = "/largeDataVolume/LeHou_GBMLGG_300x300_adaptive_ppi_gridwise_random_1/train"
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
f = open('features/'+dbname+'_macroArch'+suffix+'_valset_features.txt', 'w')
c=0
directory = "/largeDataVolume/LeHou_GBMLGG_300x300_adaptive_ppi_gridwise_random_1/test"
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
        _, index = torch.max(out, 1)
        
        patchname = os.path.basename(filename).split('.')[0]
#        y_val_pred_dict[patchname] = (int(dirname), index.item())
        print("{}, {}, {}".format(patchname, dirname, str(out.tolist()).strip('[]'))) # patchname, ground_truth, predicted label, softmax
        f.write("{}, {}, {}\n".format(patchname, dirname, str(out.tolist()).strip('[]')))
                            
#        percentage = torch.nn.functional.softmax(out, dim=1)[0]
#        class0.append(percentage[0].tolist())
#        class1.append(percentage[1].tolist())

f.close()


