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
suffix = '_inceptionv3_lr_rop_epoch_50_data_aug'
#learning_rate=0.001
classes = [0, 1, 2, 3, 4, 5]
#num_epoch=500     
num_classes = len(classes)

test_transform = transforms.Compose(
    [#transforms.Resize(299),
     transforms.CenterCrop(299),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#2. Define a Convolutional Neural Network
#take 3-channel images (instead of 1-channel images as it was defined).

from torchvision import models
from torchvision.models.inception import  BasicConv2d, InceptionA, InceptionB, \
    InceptionC, InceptionD, InceptionE, InceptionAux

class MyInceptionFeatureExtractor(nn.Module):
    def __init__(self, inception, aux_logits=False, transform_input=False):
        super(MyInceptionFeatureExtractor, self).__init__()
        inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]        
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)                
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
#        if aux_logits:
        self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        # stop where you want, copy paste from the model def

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[0] = x[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[1] = x[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[2] = x[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)        
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
         # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)        
        # copy paste from model definition, just stopping where you want
        return x

inception = models.inception_v3()
net = MyInceptionFeatureExtractor(inception)
#net = models.inception_v3()
net.aux_logits=False
net.AuxLogits.fc = nn.Linear(768, num_classes)
net.fc = nn.Linear(2048, num_classes)

# Rail_macroArch_LeHou_Large_lr_0.001_epoch_500_net.pth
#PATH = './'+dbname+'_macroArch'+suffix+'_lr_'+str(learning_rate)+'_epoch_'+str(num_epoch)+'_net.pth'
PATH = 'models/inceptionv3_imagenet_gbmlgg_300x300_adaptive_ppi_gridrs1_net.pth'
#checkpoint = torch.load(PATH) # load saved checkpoint model
#net.load_state_dict(checkpoint['state_dict'])
net.load_state_dict(torch.load(PATH))
net.eval()      
net.to(device)

#f = net(torch.rand(1, 3, 299, 299)) 
#print(f.shape) #utput: torch.Size([1, 2048])

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


