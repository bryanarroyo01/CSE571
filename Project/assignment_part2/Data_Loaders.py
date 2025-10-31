import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        ## Bryan Arroyo Code ##
        # self.data = pd.read_csv('saved/training_data.csv')
        ## Bryan Arroyo Code ##
        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        ## Bryan Arroyo ##
        dataItem = self.normalized_data[idx]
        x=torch.tensor(np.array(dataItem[:6]),dtype=torch.float32)
        y=torch.tensor(dataItem[-1],dtype=torch.float32)
        return {'input': x, 'label': y}
    #         self.getDataDictionary(dataItem)
    # def getDataDictionary(self, itm):
    #     return {
    #         'input':np.array(itm[:6],dtype=np.float32),
    #         'label':np.float32(itm[-1])
    #     }
        

class internal_Dataset(dataset.Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
    def __getitem__(self,index):
        return {'input':np.array(self.x[index],dtype=np.float32),'label':np.float32(self.y[index])}
        ## Bryan Arroyo ##
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.



class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        ## Bryan Arroyo Code ##
        train_data_size = int(.8*self.nav_dataset.__len__()) #Handle arbritrary number of samples and ensure 80% goes to training.
        test_data_size = self.nav_dataset.__len__()-train_data_size
        print(self.nav_dataset.__len__())

        torch.manual_seed(42) ## Get consistent random seed

        self.train_dataset, self.test_dataset = data.random_split(self.nav_dataset,[train_data_size,test_data_size])

        ##Balance data in training Data. We will use undersampling.
        x_trn = []
        y_trn = []

        for i in range(len(self.train_dataset)):
            sample = self.train_dataset[i]
            x = torch.tensor(sample['input'],dtype=torch.float32)
            y = torch.tensor(sample['label'],dtype=torch.float32)
            x_trn.append(x)
            y_trn.append(y)

        x_trn = torch.stack(x_trn) #Convert to torch tensor
        y_trn = torch.stack(y_trn)

        class_no_collision = (y_trn == 0).nonzero(as_tuple=True)[0]
        class_collision = (y_trn == 1).nonzero(as_tuple=True)[0]

        #Determine minority class
        minority_count =  min(len(class_no_collision),len(class_collision))

        #Random sampling
        samples_no_collision = class_no_collision[torch.randperm(len(class_no_collision))[:minority_count]]
        samples_collision = class_collision[torch.randperm(len(class_collision))[:minority_count]]
        
        #Combine and shuffle
        undersamples = torch.cat([samples_collision,samples_no_collision])
        undersamples = undersamples[torch.randperm(len(undersamples))]

        #Prepare Data loaders
        x_trn_bln=x_trn[undersamples]
        y_trn_bln=y_trn[undersamples]

        self.train_dataset = internal_Dataset(x_trn_bln,y_trn_bln)
        self.train_loader = DataLoader(self.train_dataset,batch_size,shuffle=True)
        self.test_loader = DataLoader(self.test_dataset,batch_size,shuffle=True)
        
        ## Bryan Arroyo Code ##
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
        ## Bryan Arroyo Code ##
        # print(sample)
        ## Bryan Arroyo Code ##
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']
    ## Bryan Arroyo Code ##
    print('Ran with no Issues!')
    ## Bryan Arroyo Code ##


if __name__ == '__main__':
    main()
