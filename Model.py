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
        self.local_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,6), padding=(0, 2))
        
        # 64 input channels (check this?), feature maps from local convolution,
        # 128 output channels, 20x2 kernel (check this?)
        self.global_conv=nn.Conv2d(64,128,(88,2))
        
        #LSTM layer,48 cells each
        self.dec=nn.LSTM(128,48,2,dropout=0.25)
        
        # Size of output of LSTM, for now use # of hiden state features
        self.denseFF=nn.Linear(48,7)
        self.sm=nn.LogSoftmax()
        
    def forward(self, x):
        # Apply ReLu units to the results of convolution, local convoltion layer
        x=x.float()
        x=F.relu(self.local_conv(x))
        x = nn.ZeroPad2d((0,1,0,0))(x)
        x=nn.MaxPool2d(kernel_size=(1,4))(x)
        #Global convolution layer
        x=F.relu(self.global_conv(x))
        x = nn.ZeroPad2d((0,1,0,0))(x)
        x=nn.MaxPool2d(kernel_size=(1,2))(x)
        # remove second dimension
        # x=torch.squeeze(input=x, dim=0)
        x = x.permute(3, 0, 2, 1)
        x = torch.squeeze(x, dim=2)
        out,hidden=self.dec(x)
        # Feed output through dense dense/feedforward layer with softmax activation units to
        # classify the input onto one of the 7 emotion categories.
        out = out[-1, :, :]
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
        num_speakers = 250
        self.emotions_frame = pd.read_csv(csv_file_path,header=None)
        self.transform = transform
        features = self.emotions_frame.iloc[:, 1:-1].as_matrix()
        labels =self.emotions_frame.iloc[:,-1]
        speakers=self.emotions_frame.iloc[:,0]
        speaker_array = [""]*num_speakers
        # features = features.astype('double').reshape(88)
        # sample = {'speaker': speaker, 'label': label,'features':features}
        # data=EmotionDataset(train_features)
        j=0
        num_features = len(features)
        data_array=np.zeros((num_speakers,88,512),dtype='double')
        label2index = {
        "anger":0,
        "boredom":1,
        "disgust":2,
        "fear":3,
        "happiness":4,
        "sadness":5,
        "neutral":6
        }
        label_array=['']*num_speakers
        for i in range(num_speakers):
            initialID= speakers[j]
            speaker_array[i] = initialID
            temp_array= features[j, :]
            temp_array=np.reshape(temp_array,(88,1))
            j+=1
            # new_label = np.zeroes(7)
            idx = label2index[labels[j]]
            # new_label[idx] = 1
            label_array[i]= idx
            while j < num_features and speakers[j]==initialID:
                temp_array = np.hstack((temp_array,np.reshape(features[j, :],(88,1))))
                j+=1
            if temp_array.shape[1]<512:
                pad_length = 512-temp_array.shape[1]
                temp_array = np.pad(temp_array,((0, 0), (0, pad_length)),'constant')
            elif temp_array.shape[1]>512:
                temp_array=temp_array[:,:512]
            data_array[i]=temp_array
        self.features = data_array
        self.labels = label_array
        self.speakers = speaker_array

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # features = self.emotions_frame.iloc[idx, 1:-1].as_matrix()
        # label=self.emotions_frame.iloc[idx,-1]
        # speaker=self.emotions_frame.iloc[idx,0]
        # features = features.astype('double').reshape(88)
        features = self.features[idx, :, :].astype("double")
        features = transforms.ToTensor()(features)
        speaker = self.speakers[idx]
        label = self.labels[idx]
        sample = {'speaker': speaker, 'label': label,'features':features}
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    


# In[59]:



##Extract a couple of vectors to train on
data=EmotionDataset(train_features)
# j=0
# data_array=np.zeros((5000,88,512),dtype='double')
# label_array=['']*5000
# i=0
# print(data.emotions_frame.head())
# speaks = set(data.emotions_frame.iloc[:, 0])
# print(len(speaks))
# print(speaks)
# print(data.emotions_frame.head())
# print(len(data))
# for i in range(250):
#     print(i)
#     info=data[j]
#     initialID=info['speaker']
#     temp_array=info['features']
#     temp_array=np.reshape(temp_array,(88,1))
#     j+=1
#     info=data[j]
#     label_array[i]=info["label"]
#     while info['speaker']==initialID:
#         temp_array = np.hstack((temp_array,np.reshape(info['features'],(88,1))))
#         j+=1
#         info=data[j]
#     if temp_array.shape[1]<512:
#         pad_length = 512-temp_array.shape[1]
#         temp_array = np.pad(temp_array,((0, 0), (0, pad_length)),'constant')
#     elif temp_array.shape[1]>512:
#         temp_array=temp_array[:,:512]
#     data_array[i]=temp_array
# print("NUM", j)


# In[42]:

# In[81]:


## Code to train 
# label2index = {
#         "anger":0,
#         "boredom":1,
#         "disgust":2,
#         "fear":3,
#         "happiness":4,
#         "sadness":5,
#         "neutral":6
#     }
data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=10, shuffle=False)
model=Net()
loss_fn = torch.nn.NLLLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
running_loss = 0
for epoch in range(1):
    for sample in data_loader:
        features = sample["features"]
        label = torch.tensor(sample["label"])
        # features=torch.from_numpy(sample)
        # features=torch.unsqueeze(features,0)
        # features=torch.unsqueeze(features,0)
        y_pred = model(features)
        print(y_pred)
        loss=loss_fn(y_pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # print statistics
        # running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

print('Finished Training')

