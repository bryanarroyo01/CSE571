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
        self.normalized_data = self.scaler.fit_transform(
            self.data)  # fits and transforms
        # save to normalize at inference
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb"))

    def __len__(self):
        # STUDENTS: __len__() returns the length of the dataset
        ## Bryan Arroyo Code ##
        return len(self.data)
        ## End Bryan Arroyo Code ##

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        ## Bryan Arroyo ##
        dataItem = self.normalized_data[idx]
        x = torch.tensor(np.array(dataItem[:6]), dtype=torch.float32)
        y = torch.tensor(dataItem[-1], dtype=torch.float32)
        return {'input': x, 'label': y}
    #         self.getDataDictionary(dataItem)
    # def getDataDictionary(self, itm):
    #     return {
    #         'input':np.array(itm[:6],dtype=np.float32),
    #         'label':np.float32(itm[-1])
    #     }


class internal_Dataset(dataset.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {'input': np.array(self.x[index], dtype=np.float32), 'label': np.float32(self.y[index])}
        ## Bryan Arroyo ##
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        ## Bryan Arroyo Code ##
        self.train_loader = {}
        self.test_loader = {}
        print(f"Dataset size: {self.nav_dataset.__len__()} samples")
        ## End Bryan Arroyo Code ##
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
