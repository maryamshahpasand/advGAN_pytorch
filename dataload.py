from random import shuffle

import numpy as np
import sys
import psutil
import os
import csv
import scipy.sparse
from  sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset,sampler
import torch
import random
batch_size = 128
def load_data(train=True):
    features = np.load('./MalwareDataset/Drebin_important_features.npy')
    features=np.append(features,[features[len(features)-1]],axis=0)
    x = scipy.sparse.load_npz('./MalwareDataset/x_train.npz' ).toarray()
    x_ = scipy.sparse.load_npz('./MalwareDataset/x_malware_binary_features_with_families.npz' ).toarray()
    y = np.load('./MalwareDataset/y_train.npy' )
    x_mal_train = x[np.where(y == 1)]
    x_ben_train = x[np.where(y == 0)]
    # x_ben_train = x_ben_train[random.sample(range(x_ben_train.shape[0]), x_mal_train.shape[0])]
    #
    # train_data = [(x_mal, y_mal), (x_ben, y_ben)]
    # sampling for unbalanced data
    # class_sample_count = np.array(
    #     [len(np.where(y == t)[0]) for t in np.unique(y)])
    # weight = 1. / class_sample_count
    # samples_weight = []
    # for t in range(len(y) - 1):
    #     samples_weight.append(weight[int(y[t])])
    # Sampler = sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    # data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    # data_loader = DataLoader(data, batch_size=batch_size, sampler=Sampler, drop_last=True)

    x = scipy.sparse.load_npz('./MalwareDataset/x_test.npz').toarray()
    y = np.load('./MalwareDataset/y_test.npy')
    x_mal = x[np.where(y == 1)]
    x_ben = x[np.where(y == 0)]
    x_ben = x_ben[0:len(x_mal)]
    y_mal = np.ones(x_mal.shape[0])
    y_ben = np.zeros(x_ben.shape[0])
    x = np.concatenate([x_mal, x_ben])
    y = np.concatenate([y_mal, y_ben])
    test_data = (x,y)

    # xmal=np.concatenate([x_mal, x_mal_train])
    # x_mal_train,xmal_test,y_xmaltrain,y_xmaltest = train_test_split(xmal,np.ones(xmal.shape[0]),shuffle=True,test_size=0.2)
    #
    #
    # xben = np.concatenate([x_ben, x_ben_train])
    # x_ben_train,xben_test,y_xbentrain,y_xbentest = train_test_split(xben,np.zeros(xben.shape[0]),shuffle=True,test_size=0.2)


    # test_data = (np.concatenate([xmal_test,xben_test]),np.concatenate([y_xmaltest,y_xbentest]))
    return (x_mal_train,x_ben_train) ,test_data,  x.shape[1],features

def dataloader(x_mal ,x_ben):
    y_mal = np.ones(x_mal.shape[0])
    y_ben = np.zeros(x_ben.shape[0])
    x = np.concatenate([x_mal, x_ben])
    y = np.concatenate([y_mal, y_ben])
    class_sample_count = np.array(
        [len(np.where(y == t)[0]) for t in np.unique(y)])
    weight = 1. / class_sample_count
    samples_weight = []
    for t in range(len(y) - 1):
        samples_weight.append(weight[int(y[t])])
    Sampler = sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    data_loader = DataLoader(data, batch_size=batch_size, sampler=Sampler, drop_last=True)
    return data_loader

# if __name__ == "__main__":