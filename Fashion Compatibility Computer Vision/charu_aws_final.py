#!/usr/bin/env python
# coding: utf-8

# In[10]:


#!/usr/bin/env python
# coding: utf-8

# In[4]:


# utils.py

import numpy as np
import os
import os.path as osp
import argparse

Config ={}
Config['root_path'] = "./polyvore_outfits/"
Config['meta_file'] = "polyvore_item_metadata.json"
Config['test_file'] = "test_category_hw.txt"
Config['checkpoint_path'] = ''
#Config['train_compatibility']='pairwise_compatibility_train.txt'
#Config['valid_compatibility']='pairwise_compatibility_valid.txt'



Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 20
Config['batch_size'] = 64

Config['learning_rate'] = 0.001
Config['num_workers'] = 5 #aws might not need it, original value = 5


# In[2]:


# data.py

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

#from utils import Config


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()



    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms



    def create_dataset(self):
        # map id to category
        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        id_to_category = {}
        for k, v in tqdm(meta_json.items()):
            id_to_category[k] = v['category_id']

        # create X, y pairs
        files = os.listdir(self.image_dir)
        X = []; y = []
        for x in files:
            if x[:-4] in id_to_category:
                X.append(x)
                y.append(int(id_to_category[x[:-4]]))

        y = LabelEncoder().fit_transform(y)
        print('len of X: {}, # of categories: {}'.format(len(X), max(y) + 1))

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, max(y) + 1



# For category classification
class polyvore_train(Dataset):
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_train[item])
        return self.transform(Image.open(file_path)),self.y_train[item]




class polyvore_test(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')


    def __len__(self):
        return len(self.X_test)


    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_test[item])
        return self.transform(Image.open(file_path)), self.y_test[item]




def get_dataloader(debug, batch_size, num_workers):
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, classes = dataset.create_dataset()

    if debug==True:
        train_set = polyvore_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = polyvore_test(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    else:
        train_set = polyvore_train(X_train, y_train, transforms['train'])
        test_set = polyvore_test(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    datasets = {'train': train_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'test']}
    return dataloaders, classes, dataset_size


def category_text_generation():
        meta_file = open(osp.join(Config['root_path'], Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        id_to_category = {}
        m = 0
        for k, v in tqdm(meta_json.items()):
            id_to_category[k] = v['category_id']
     
        # create X, y pairs
        files = os.listdir(osp.join(Config['root_path'], 'images'))
        y = []
        for x in files:
            if x[:-4] in id_to_category:
                y.append(int(id_to_category[x[:-4]]))
        le = LabelEncoder()
        g = le.fit_transform(y)
        B = []
        f = open(Config['root_path'] + "test_category_hw.txt", "r")
        a = [line.split() for line in f.readlines()]
        for i in range(len(a)):
            a[i][0] = a[i][0] + '.jpg'
            B.append(a[i][0])

        a = open(Config['root_path'] + "test_category_hw.txt", "r")
        b = open('Charu_Model_test_category_hw.txt', 'w')

        f = [lines.split() for lines in a.readlines()]
        J = []
        for i in range(len(f)):
            J.append(f[i][0])


        dataset = polyvore_dataset()
        transforms = dataset.get_data_transforms()['test']

        size = int(np.floor(len(B) / Config['batch_size']))
        
        #model_copy_tensor = torch.load('./Results/ResNet.pth')
        check_point = torch.load('model.pth')
        model_copy = check_point['model']
        #model_copy = load_model('Build_model.hdf5')
        #model_copy.eval()
        
        for i in range(0, size*Config['batch_size'], Config['batch_size']):
            X= []
            Y =[]
            ans = []
            for j in range(Config['batch_size']):
                file_path = osp.join(osp.join(Config['root_path'], 'images'), B[i+j])
                l = transforms(Image.open(file_path))
                X.append(l)
                Y.append(id_to_category[J[i+j]])
                
            #Y = np.stack(C)
            #Y = np.moveaxis(Y, 1, 3)
            
            with torch.no_grad():
                for inputs in X:
                    # print(inputs.shape)
                    inputs = inputs.to(device)
                    outputs = model_copy(inputs[None,...])
                    _, pred = torch.max(outputs,1)
                    for p in pred:
                        ans.append(p)
            
            
#             acc, loss1, ans = eval_model(model_copy_tensor, Y, criterion, device)
#             # ans = (model_copy.predict(Y))

#             for k in range(len(ans)):
#                 ans1.append(np.argmax(ans[k]))
            preds = le.inverse_transform(np.asarray(ans, dtype= np.int32))
            for p in range(Config['batch_size']):
                b.write(J[p+m] + '\t' + str(preds[p]) + '\t' + id_to_category[J[p+m]] + '\n')
            m = m + Config['batch_size']
            if m == size*Config['batch_size']:
                break
        b.close()

########################################################################
# For Pairwise Compatibility Classification


# In[3]:


# model.py

# class MyMobileNet(nn.Module):
#     def __init__(self, my_pretrained_model):
#         super(MyMobileNet, self).__init__()
#         self.pretrained = my_pretrained_model
#         self.my_new_layers = nn.Sequential(
#                                            nn.Dropout(0.4),
#                                            nn.Linear(1000, 200),
#                                            nn.ReLU(),
#                                            nn.Linear(200, 153)
#                                            )

#     def forward(self, x):
#         x = self.pretrained(x)
#         x = self.my_new_layers(x)
#         return x

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, conv1_dim=32, conv2_dim=64, conv3_dim=128, conv4_dim=256, conv5_dim=512):
        super(Net, self).__init__()
        self.conv5_dim = conv5_dim

        self.conv1 = nn.Conv2d(3, conv1_dim, 5, stride=1, padding=2)
        self.conv2= nn.Conv2d(32, 32, 5, stride=1, padding=2)
        
        self.conv3 = nn.Conv2d(conv1_dim, conv2_dim, 3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(conv2_dim, conv2_dim, 3, stride=1, padding=2)
        
        
        self.conv5 = nn.Conv2d(conv2_dim, conv3_dim, 3, stride=1, padding=2)
        self.conv6 = nn.Conv2d(conv3_dim, conv3_dim, 3, stride=1, padding=2)
        
        
        self.conv7 = nn.Conv2d(conv3_dim, conv4_dim, 3, stride=1, padding=2)
        self.conv8 = nn.Conv2d(conv4_dim, conv4_dim, 3, stride=1, padding=2)
        
        self.conv9 = nn.Conv2d(conv4_dim, conv5_dim, 3, stride=1, padding=2)
        self.conv10 = nn.Conv2d(conv5_dim, conv5_dim, 3, stride=1, padding=2)
        
        
       

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1=nn.Dropout(0.1)
        self.dropout2=nn.Dropout(0.2)
        self.dropout3=nn.Dropout(0.3)

        self.fc1 = nn.Linear(conv5_dim * 10 * 10, 1000) # 3x3 is precalculated and written, you need to do it if you want to change the # of filters
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 153)

        self.normalize1 = nn.BatchNorm2d(conv1_dim)
        self.normalize2 = nn.BatchNorm2d(conv1_dim)
        self.normalize3 = nn.BatchNorm2d(conv2_dim)
        self.normalize4 = nn.BatchNorm2d(conv2_dim)
        self.normalize5 = nn.BatchNorm2d(conv3_dim)
        self.normalize6 = nn.BatchNorm2d(conv3_dim)
        self.normalize7 = nn.BatchNorm2d(conv4_dim)
        self.normalize8 = nn.BatchNorm2d(conv4_dim)
        self.normalize9 = nn.BatchNorm2d(conv5_dim)
        self.normalize10 = nn.BatchNorm2d(conv5_dim)

    def forward(self, x):
        x = F.relu(self.normalize1((self.conv1(x))))
        x = self.pool(F.relu(self.normalize2((self.conv2(x))))) # first convolutional then batch normalization then relu then max pool
        
        x = F.relu(self.normalize3((self.conv3(x))))
        x = self.pool(F.relu(self.normalize4((self.conv4(x)))))
        x=self.dropout2(x)
        
        x = F.relu(self.normalize5((self.conv5(x))))
        x = self.pool(F.relu(self.normalize6((self.conv6(x)))))
        
          
        x = F.relu(self.normalize7((self.conv7(x))))
        x = self.pool(F.relu(self.normalize8((self.conv8(x)))))
        x=self.dropout2(x)
        
        x = F.relu(self.normalize9((self.conv9(x))))
        x = self.pool(F.relu(self.normalize10((self.conv10(x)))))
        #print(x.shape)
        x = x.view(-1, self.conv5_dim * 10 * 10) # flattening the features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x=self.dropout3(x)
        x = self.fc3(x)

        return x
model = Net()





# In[ ]:


# train_category.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt

#from utils import Config
#from model import model
#from data import get_dataloader



def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):
    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_list= []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred==labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            if phase == 'train':
                train_loss_list.append(epoch_loss)
                train_acc_list.append(epoch_acc)
                
            if phase == 'test':
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase=='test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model)

        #torch.save({'model':best_model_wts}, osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth'))
        #print('Model saved at: {}'.format(osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth')))
        torch.save({'model':best_model_wts}, 'model.pth')
        print('Model saved at: {}'.format('model.pth'))

    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))

    plt.figure()
    plt.plot(np.arange(num_epochs),train_loss_list,label='Train')
    plt.plot(np.arange(num_epochs),val_loss_list, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('./Charu_new_loss.png', dpi=256)
    #plt.show()
    
    plt.figure()
    plt.plot(np.arange(num_epochs),train_acc_list,label='Train')
    plt.plot(np.arange(num_epochs),val_acc_list, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('./Charu_new_acc.png', dpi=256)
    #plt.show()

if __name__=='__main__':

    dataloaders, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=Config['learning_rate'],weight_decay=0.0001)
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

    train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size)
    category_text_generation()

    print('DONE')

