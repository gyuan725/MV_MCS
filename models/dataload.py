# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:36:18 2023

@author: dell
"""
import torch
import pickle

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def get_path(root, dataset):
    path = '{}{}/{}'.format(root, dataset, dataset)
    return path


def MV_data(root, dataset, gama, cols):
    
    path = get_path(root, dataset)
    multi_df = pd.read_csv(path + ".csv")
    y = multi_df['Class'].values
    y = y.astype(int)
    if min(y)==1:
        y = y-1
    X = []
    V = []
    for i in range(len(cols)):
        col = cols[i]
        df = multi_df[col]
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        X.append(df_scaled[col].values)
        # X.append(df.values)
        v = preprocess_data(X[i], gama)
        V.append(v)
    data = {"X": X, "V": V, "y": y}
    with open(path + str(gama) + ".pkl", "wb") as f:
        pickle.dump(data, f)
    
    return X, V, y

class MultiViewDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        # 假设每个视角的数据行数相同
        return len(self.y)
    
    def __getitem__(self, idx):
        # 获取每个视角对应的那一行数据
        data = []
        for v_num in range(len(self.X)):
            data.append((self.X[v_num][idx]).astype(np.float32))

        label = self.y[idx]
        return data, label

def preprocess_data(X,gama):
    num_g = X.shape[1]
    num_n = X.shape[0]
    z = np.zeros([num_g, gama+1])
    for i in range(num_g):
        for j in range(gama+1):
            z[i,j] = min(X[:,i]) + j / gama * (max(X[:,i]) - min(X[:,i]))
            # z[i,j] =  j / gama 
            # z[i,j] = 0 + j / gama * 100
    v = np.zeros([num_n, num_g, gama])
    for i in range(num_n):
        for j in range(num_g):
            for k in range(gama):
                v[i,j,k] = np.where( z[j,k] <= X[i,j] <= z[j,k+1], (X[i,j]- z[j,k]) / (z[j,k+1] - z[j,k]), np.where( X[i,j] < z[j,k], 0 , 1))
    v = np.reshape(v, (num_n, num_g * gama))
    
    return v

def preprocess_data_fast(X, gama, x_min, x_max):
    X = np.asarray(X, dtype=np.float64)
    num_n, num_g = X.shape
    x_range = x_max - x_min               # (num_g,)

    # 防止某一列全相等导致除零
    safe_range = np.where(x_range == 0, 1.0, x_range)

    # 先把每个特征缩放到 [0, gama]
    s = (X - x_min) / safe_range * gama   # (num_n, num_g)

    # 若该列全相等，希望输出全0，可单独处理
    s[:, x_range == 0] = 0.0
    # k = 0, 1, ..., gama-1
    k = np.arange(gama, dtype=np.float64)  # (gama,)

    # 广播:
    # s[..., None]  -> (num_n, num_g, 1)
    # k             -> (gama,)
    # 结果          -> (num_n, num_g, gama)
    v = np.clip(s[..., None] - k, 0.0, 1.0)
    v = v.reshape(num_n, num_g * gama)
    return v

def preprocess_data_mid(X, gama):
    X = np.asarray(X, dtype=np.float64)
    num_n, num_g = X.shape

    x_min = X.min(axis=0)
    x_max = X.max(axis=0)

    z = x_min[:, None] + (np.arange(gama + 1) / gama) * (x_max - x_min)[:, None]

    dz = z[:, 1:] - z[:, :-1]
    dz = np.where(dz == 0, 1.0, dz)

    v = np.clip((X[:, :, None] - z[:, :-1][None, :, :]) / dz[None, :, :], 0.0, 1.0)
    v = v.reshape(num_n, num_g * gama)

    return v


def preprocess_uta_data(X, gama, scale):
    num_g = X.shape[1]
    num_n = X.shape[0]
    z = np.zeros([num_g, gama+1])
    for i in range(num_g):
        for j in range(gama+1):
            z[i,j] = scale[i,0] + j / gama * (scale[i,1] - scale[i,0])
            # z[i,j] = 0 + j / gama * 100
    v = np.zeros([num_n, num_g, gama])
    for i in range(num_n):
        for j in range(num_g):
            for k in range(gama):
                v[i,j,k] = np.where( z[j,k] <= X[i,j] <= z[j,k+1], (X[i,j]- z[j,k]) / (z[j,k+1] - z[j,k]), np.where( X[i,j] < z[j,k], 0 , 1))
    v = np.reshape(v, (num_n, num_g * gama))
    
    return v

def preprocess_choquet_data(X):
    num_g = X.shape[1]
    row, col = list(), list()
    for i in range(num_g -1):
        for j in range(i + 1, num_g):
            row.append(i), col.append(j)
    p, q = X[:, row], X[:, col]
    v1 = np.minimum(p,q)
                
    return v1, row, col


def one_hot_embedding(y, num_class):
    # Convert to One Hot Encoding
    y = y.long()
    labels = torch.eye(num_class)
    return labels[y]


