#!/usr/bin/env/python
# coding: utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import torch.backends.cudnn as cudnn
cudnn.enabled=False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# from skimage import io, transform
import numpy as np
from numpy import newaxis
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
train_features = ['./RawTrainingFeatures1.csv','./RawTrainingFeatures2.csv']
import sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# In[80]:
original_feature_indices = [0, 10, 22, 24, 26, 28, 30, 32, 34, 35, 36, 40, 43, 52, 56, 57, 66, 76, 77, 80]
# nf = len(original_feature_indices)
nf = 88
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 64 output channels, 1x6  convolution
        # kernel
        self.local_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,6), padding=(0, 2))
        self.drop_out = nn.Dropout(p=.5)
        # 64 input channels (check this?), feature maps from local convolution,
        # 128 output channels, 20x2 kernel (check this?)
        self.global_conv=nn.Conv2d(64,128,(nf,2))
        #LSTM layer,48 cells each
        self.dec=nn.LSTM(128,48,2,dropout=0.5)
        
        # Size of output of LSTM, for now use # of hiden state features
        self.denseFF=nn.Linear(48,7)
        self.sm=nn.LogSoftmax()
        
    def forward(self, x):
        # Apply ReLu units to the results of convolution, local convoltion layer
        x=x.float()
        x = self.drop_out(x)
        x=F.relu(self.local_conv(x))
        x = self.drop_out(x)
        x = nn.ZeroPad2d((0,1,0,0))(x)
        x=nn.MaxPool2d(kernel_size=(1,4))(x)
        #Global convolution layer
        x=F.relu(self.global_conv(x))
        x = self.drop_out(x)
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

#initial_data = pd.read_csv(train_features,header=None)


# In[5]:


# print(initial_data.head())


# In[58]:


class EmotionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csvs, transform=None,test=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        csv_file_path_1=csvs[0]
        csv_file_path_2=csvs[1]
        emotions_frame1 = pd.read_csv(csv_file_path_1,header=None)
        emotions_frame2 = pd.read_csv(csv_file_path_2,header=None)
        self.emotions_frame=pd.concat([emotions_frame1,emotions_frame2],ignore_index=True)
#         self.emotions_frame=emotions_frame2
#         print(self.emotions_frame)
#         self.emotions_frame=emotions_frame1
#         print(self.emotions_frame)
#         self.emotions_frame,self.test_frames=train_test_split(self.emotions_frame,test_size=0.1)
#         if not(test):
#             self.emotions_frame,self.test_frames=train_test_split(self.emotions_frame,test_size=0.1,random_state=42,shuffle=False)
#         else:
        self.test_frames=None
        self.test_labels = None
        self.test = False
        num_speakers = 535 ##self.emotions_frame.shape[0]
        self.transform = transform
        self.speaker_map={}
        features = self.emotions_frame.iloc[:, 1:-1].as_matrix()
        # features = np.take(features, original_feature_indices, axis=1)
        labels =self.emotions_frame.iloc[:,-1]
        speakers=self.emotions_frame.iloc[:,0]
        speaker_array = [""]*num_speakers
       
        j=0 
        num_features = len(features)

        data_array=np.zeros((num_speakers,nf,512),dtype='double')
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
            speaker=initialID[1:3]
            speaker_array[i] = initialID
            temp_array= features[j, :]
            temp_array=np.reshape(temp_array,(nf,1))
            j+=1
#             print(labels)
#             print(j,num_features,speakers[j],initialID)
            idx = label2index[labels[j]]
            label_array[i]= idx
           
            while j < num_features and speakers[j]==initialID:
                temp_array = np.hstack((temp_array,np.reshape(features[j, :],(nf,1))))
                j+=1
            if temp_array.shape[1]<512:
                pad_length = 512-temp_array.shape[1]
                temp_array = np.pad(temp_array,((0, 0), (0, pad_length)),'constant')
            elif temp_array.shape[1]>512:
                temp_array=temp_array[:,:512]
            data_array[i]=temp_array
            
            if speaker in self.speaker_map:
                self.speaker_map[speaker]=np.append(self.speaker_map[speaker],temp_array[newaxis,::],axis=0)
            else:
                self.speaker_map[speaker]=np.empty((1,nf,512))
                self.speaker_map[speaker][0,:,:]=temp_array
        self.features = data_array
        self.labels = label_array
        self.speakers = speaker_array
        self.std_map={}
        self.mean_map={}
        for ID in self.speaker_map.keys():
            std=np.std(self.speaker_map[ID],axis=(0,2))
            std=std[:,newaxis]
            std_zeros= std==0
            std[std_zeros]=1
            
            mean=np.mean(self.speaker_map[ID],axis=(0,2))
            mean=mean[:,newaxis]
            for j in range(511):
                std=np.insert(std,1,std[:,0],axis=1)
                mean=np.insert(mean,1,mean[:,0],axis=1)
            self.std_map[ID]=std
            self.mean_map[ID]=mean
    def get_test_csv(self):
        return self.test_frame.to_csv()
    def __len__(self):
        if self.test:
            return len(self.test_frames)
        return len(self.features)
    def filter_speakers(self, speaker):
        to_remove = []
        test_remove = []
        for i in range(len(self.speakers)):
            current_speaker = self.speakers[i][1:3]
            if current_speaker == speaker:
                to_remove.append(i)
            else:
                test_remove.append(i)
        self.test_frames = np.delete(self.features, test_remove, axis=0)
        self.test_labels = [x for i, x in enumerate(self.labels) if i not in test_remove]
        self.features = np.delete(self.features, to_remove, axis=0)
        self.labels = [x for i, x in enumerate(self.labels) if i not in to_remove]

    
    def __getitem__(self, idx):
        speaker = self.speakers[idx]
        if self.test:
            features = self.test_frames[idx, :, :].astype("double")
            label = self.test_labels[idx]
        else:
            features = self.features[idx, :, :].astype("double")
            label = self.labels[idx]
        features=(features-self.mean_map[speaker[1:3]])/self.std_map[speaker[1:3]]
        features = transforms.ToTensor()(features)
        features = features.to(device)        

        sample = {'speaker': speaker, 'label': label,'features':features}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    


# In[59]:

## Code to train 
data=EmotionDataset(train_features)

# data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=64, shuffle=False)
# model=Net()
# model.cuda()
# # model.load_state_dict(torch.load('checkpoint_test0.pth'))
# cudnn.benchmark = True
#
# loss_fn = torch.nn.NLLLoss()
# learning_rate = 1e-3
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# running_loss = 0
# print('started training')
# running_loss=0.0
# i=0
# j=0
import time
start_time = time.time()
# output_results_file = "Results.csv"
# f = open(output_results_file, "w")
first = True
for speaker in sorted(list(data.speaker_map))[1:]:
    speaker = "15"
    data=EmotionDataset(train_features)
    data.filter_speakers(speaker)
    model=Net()
    model.cuda()

    # model.load_state_dict(torch.load('checkpoint_test88_val'+str(speaker)+'.pth'))
    cudnn.benchmark = True

    loss_fn = torch.nn.NLLLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    running_loss = 0
    print('started training')
    running_loss=0.0
    i=0
    j=0
    best_testing_accuracy = 0
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=64, shuffle=True)
    for epoch in range(700):
        total = 0
        correct = 0
        total_test = 0
        correct_test = 0
        model.train()
        for sample in data_loader:
            features = sample["features"]
            label = torch.tensor(sample["label"])
            y_pred = model(features)
            loss=loss_fn(y_pred,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            i+=64

            _,predicted=torch.max(y_pred.data,1)
            total+=label.size(0)
            correct+=(predicted==label).sum().item()
        data.test = True
        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=64, shuffle=False)
        model.eval()

        for sample in data_loader:
            features = sample["features"]
            test_pred = model(features)
            test_labels = torch.tensor(sample["label"])
            _ ,test_predicted=torch.max(test_pred.data,1)
            total_test+=test_labels.size(0)
            correct_test+=(test_predicted==test_labels).sum().item()
            # for i in range(test_labels.size(0)):
            #     if test_labels[i] != test_predicted[i]:
            #         print(test_labels[i].data, test_predicted[i].data)
        data.test = False


        if i>=100:
            i=0
            j+=1
            print('[%d,%5d] loss:%.3f' % (epoch+1,j,running_loss/1000))
            running_loss=0
        # if epoch % 1000 == 0:
        #     torch.save( model.state_dict(),'checkpoint_test'+str(i)+'.pth')

        print("Epoch" + str(epoch) + "Finished")
        end_time = time.time()
        print("Time Elapsed" + str(end_time - start_time))
        print('Training Accuracy: %d %%' % (100*correct/total))
        print('Testing Accuracy: %d %%' % (100*correct_test/total_test))
        print('Best Testing Accuracy: %d %%' % (100*best_testing_accuracy))
        if correct_test/total_test >= best_testing_accuracy:
            best_testing_accuracy = correct_test/total_test
            # torch.save( model.state_dict(),'checkpoint_test88_val'+str(speaker)+'.pth')
    exit()
    # torch.save( model.state_dict(),'checkpoint_test2_val'+str(speaker)+'.pth')
# f.close()
print('Finished Training')




