#!/usr/bin/env/python
# coding: utf-8

# In[55]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
train_features = "C:\\Users\\stick\\Documents\\MITJuniorYear\\6.345\\RawTrainingFeatures1.csv"

# In[80]:


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 64 output channels, 1x6  convolution
        # kernel
        self.local_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,6))
        
        # 64 input channels (check this?), feature maps from local convolution,
        # 128 output channels, 20x2 kernel (check this?)
        self.global_conv=nn.Conv2d(64,128,(20,2))
        
        #LSTM layer,48 cells each,  how to set 0.25 dropout rate ???
        self.dec=nn.LSTM(128,48,2,dropout=0.25)
        
        # Size of output of LSTM, for now use # of hiden state features
        self.denseFF=nn.Linear(48,7)
        self.sm=nn.LogSoftmax()
        
    def forward(self, x):
        # Apply ReLu units to the results of convolution, local convoltion layer
        x=x.float()
        x=F.relu(self.local_conv(x))
        print(x.size())
        x=nn.MaxPool2d(1,4)(x)
        print("--------")
        print(x.size())
        #Global convolution layer
        x=F.relu(self.global_conv(x))
        x=nn.MaxPool2d(1,2)(x)
        print(x.size())
        x=torch.squeeze(x)
        print(x.size())
        out,hidden=self.dec(x)
        
        # Feed output through dense dense/feedforward layer with softmax activation units to 
        # classify the input onto one of the 7 emotion categories.
        out=self.sm(self.denseFF(out))
        return out
        




# In[72]:


# training 

initial_data = pd.read_csv(train_features,header=None)


# In[5]:


# print(initial_data.head())


# In[58]:


class EmotionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.emotions_frame = pd.read_csv(csv_file_path,header=None)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        features = self.emotions_frame.iloc[idx, 1:-1].as_matrix()
        label=self.emotions_frame.iloc[idx,-1]
        speaker=self.emotions_frame.iloc[idx,0]
        features = features.astype('double').reshape(88)
        sample = {'speaker': speaker, 'label': label,'features':features}
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    


# In[59]:



##Extract a couple of vectors to train on
data=EmotionDataset(train_features)
j=0
data_array=np.zeros((50,88,512),dtype='double')
label_array=['']*50
for i in range(50):
    info=data[j]
    initialID=info['speaker']
    temp_array=info['features']
    temp_array=np.reshape(temp_array,(88,1))
    j+=1
    info=data[j]
    label_array[i]=initialID
    while info['speaker']==initialID:
        temp_array = np.hstack((temp_array,np.reshape(info['features'],(88,1))))
        j+=1
        info=data[j]
    if temp_array.shape[1]<512:
        pad_length = 512-temp_array.shape[1]
        temp_array = np.pad(temp_array,((0, 0), (0, pad_length)),'constant')
    elif temp_array.shape[1]>512:
        temp_array=temp_array[:,:512]
    data_array[i]=temp_array
        
    
                                    
    
        
        


# In[42]:


print(data_array.shape)


# In[81]:


## Code to train 
label2index = {
        "anger":0,
        "boredom":1,
        "disgust":2,
        "fear":3,
        "happiness":4,
        "sadness":5,
        "neutral":6
    }

model=Net()
loss_fn = torch.nn.NLLLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1):
    for i in range(50):
        sample=data_array[1,:,:]
        features=torch.from_numpy(sample)
        features=torch.unsqueeze(features,0)
        # features=torch.unsqueeze(features,0)
        print(features.size())
        y_pred=model(features)
        loss=loss_fn(y_pred,label2index[label_array[i]])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

