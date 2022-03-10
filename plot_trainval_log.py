# run with Python 3.5.x + 
import matplotlib.pyplot as plt
import numpy as np
import sys

model = sys.argv[1] #'resnet50' #'inception_v3'#'vgg16' #'vgg19bn' #'vgg16' #'inception_v3'#'resnet50'

src = 'log/finetune_'+model+'_cancer_data_aug.log'
f = open(src)
content = f.readlines()
f.close()

epochs,train_loss, train_acc, val_loss, val_acc = [],[],[],[],[]
for i in range(len(content)):
    line = content[i]
    if 'Epoch' in line:
        epoch_num = line.split()[1].split('/')[0] 
        epochs.append(epoch_num)
        tline = content[i+2]
        parts = tline.split()
        tloss = float(parts[2])
        tacc = float(parts[4])
        vline = content[i+3]
        parts = vline.split()
        vloss = float(parts[2])
        vacc = float(parts[4])        
        train_loss.append(tloss)
        train_acc.append(tacc)
        val_loss.append(vloss)
        val_acc.append(vacc)

plt.plot(epochs, train_acc, label='train acc')
plt.plot(epochs, val_acc, label='val acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title(model+' trainval accuracy for GBMLGG 100ppi random dataset')
plt.xlim([0, len(epochs)])
plt.xticks(np.arange(0, int(epoch_num), step=5))
plt.ylim([0, 1])
plt.legend()
plt.savefig('plots/trainval_acc_'+model+'.png')
plt.show()

plt.figure()
plt.plot(epochs, train_loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(model+' trainval loss for GBMLGG 100ppi random dataset')
plt.xlim([0, len(epochs)])
plt.xticks(np.arange(0, int(epoch_num), step=5))
plt.ylim([0, 4])
plt.legend()
plt.savefig('plots/trainval_loss_'+model+'.png')
plt.show()       

#print(train_acc)
#print(val_acc)
#print(train_loss)
#print(val_loss) 
