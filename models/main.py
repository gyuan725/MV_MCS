# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:36:32 2024

@author: gyuan
"""
import config
import numpy as np
import pickle
import pandas as pd
import torch
from dataload import MV_data, MultiViewDataset, preprocess_data_fast
import argparse
from sklearn.model_selection import StratifiedKFold

from torch.utils.data import TensorDataset, DataLoader
from train import train, test_noise


torch.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CCA', help='which dataset to use')
parser.add_argument('--gama', type=int, default=4, help='number of piecewise ponit')
parser.add_argument('--emb_dim', type=int, default=2, help='dimension of latent vector')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--batch_size', type=int, default = 16, help='batch size')
parser.add_argument('--num_epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('--random_seed', type=int, default=2024, help='random_seed')
parser.add_argument('--multi_view', type=bool, default= True, help='if multi_view')
parser.add_argument('--ER', type=bool, default= True, help='if ER fusion')
parser.add_argument('--EPL', type=bool, default = False, help='if uncertain estimation')
args = parser.parse_args()

file_path = f"../data/{args.dataset}/{args.dataset}{args.gama}.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)
    X = data['X']
    V = data["V"]
    y = data["y"]
    
X, V, y = MV_data('../data/', args.dataset, args.gama, config.cols[args.dataset])

num_t = max(y)##label from 0


    

def generate_noise(data, mean, std, seed=args.random_seed):
    np.random.seed(seed)  # 设置随机种子
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def cross_test_noise( args, X, V, y, flag, seed):
    
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state= seed) 
    noises = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    metrics = np.zeros([len(noises)+1,5,4])
    df = pd.DataFrame()
    records = pd.DataFrame()
    results = []
    num_t = max(y)

    
    for i, (train_idx, test_idx) in enumerate(skf.split(X[0], y)):
        
        X_train, V_train, y_train = [arr[train_idx] for arr in X], [arr[train_idx] for arr in V], y[train_idx]             
        X_test, V_test, y_test = [arr[test_idx] for arr in X], [arr[test_idx] for arr in V], y[test_idx]
        X_test_noise = X_test.copy()
        V_test_noise = V_test.copy()
               
        for j in range(len(noises)):
            X_test_noise[flag[i]] = generate_noise(X_test[flag[i]], mean=0, std=noises[j], seed=args.random_seed)
            # V_test_noise[flag[i]] = preprocess_data(X_test_noise[flag[i]], args.gama)
            x_min, x_max = X[flag[i]].min(axis=0), X[flag[i]].max(axis=0)
            V_test_noise[flag[i]] = preprocess_data_fast(X_test_noise[flag[i]], args.gama, x_min, x_max)
            
            if args.multi_view:
                num_g = [X[view].shape[1] for view in range(len(X))]
                noise_test_dataset = MultiViewDataset(V_test_noise, y_test)
            else:
                X1 = np.concatenate(X, axis=1)
                num_g = X1.shape[1]
                X_test_noise_cat=np.concatenate(V_test_noise, axis=1)
                noise_test_dataset = TensorDataset(torch.tensor(X_test_noise_cat).float(), torch.tensor(y_test).float())
            
            noise_test_loader = DataLoader(noise_test_dataset, batch_size=args.batch_size, shuffle= True)
            noise_metrics, noise_records = test_noise(args, num_g, num_t, noise_test_loader, fold = i, seed =seed)
            metrics[j+1, i, :] = noise_metrics
            results.append({
            'seed': seed,
            'Fold': i + 1,
            'Noise': noises[j],
            'metrics':noise_metrics,
            'records':noise_records })
        results_df = pd.DataFrame(results)
          
    return metrics, df, results, results_df  

def cross_test( args, X, V, y, seed):
    cv = 5
    skf = StratifiedKFold(n_splits=cv, shuffle = True, random_state= seed) 
    metrics = np.zeros([cv,4])
    df = pd.DataFrame()
    records = pd.DataFrame()
    num_t = max(y)
    
    
    for i, (train_idx, test_idx) in enumerate(skf.split(X[0], y)):
        
        X_train, V_train, y_train = [arr[train_idx] for arr in X], [arr[train_idx] for arr in V], y[train_idx] 
        X_test, V_test, y_test = [arr[test_idx] for arr in X], [arr[test_idx] for arr in V], y[test_idx]
              
        if args.multi_view:
            num_g = [X[i].shape[1] for i in range(len(X))]
            train_dataset = MultiViewDataset(V_train, y_train)
            test_dataset = MultiViewDataset(V_test, y_test)
            
        else:
            X1 = np.concatenate(X, axis=1)
            num_g = X1.shape[1]
            X_train,X_test=np.concatenate(V_train, axis=1),np.concatenate(V_test, axis=1)
            
            # X1 = X[0]
            # num_g = X1.shape[1]
            # X_train,X_test=V_train[0],V_test[0]
        
            train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
            test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle= True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle= True)
        df_fold, test_metrics, test_records = train(args, num_g, num_t, train_loader, test_loader, fold = i)
        metrics[i,:] = test_metrics
        records = pd.concat([records, test_records])
        df = pd.concat([df, df_fold])
        
    
    return metrics, df, records  

def repeat_CV(args, X, V, y):
    
    seeds = [0, 1, 2, 3, 4]
    repeat_metrics = []
    repeat_df = []
    repeat_records = []
    for seed in seeds:
        args.random_seed = seed
        metrics, df, records = cross_test( args, X, V, y, seed)
        repeat_metrics.append(metrics)
        repeat_df.append(df)
        repeat_records.append(records)
    
    return repeat_metrics, repeat_df, repeat_records

def noise_result(args, X, V, y):
    seeds = [0, 1, 2, 3, 4]
    noise_results = {}
    noise_metrics = {}
    noise_mean_metrics = {}
    noise_std_metrics = {}
    model_name = ['er', 'ds', 'epl', 'pl']
    flags = [
    {"multi_view": True, "ER": True, "EPL": False},
    {"multi_view": True, "ER": False, "EPL": False},
    {"multi_view": False, "ER": False, "EPL": True},
    {"multi_view": False, "ER": False, "EPL": False}]
    noise_views = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        noise_views[seed] = rng.integers(0, len(X), size=5)

    
    i = 0
    for i, flag in enumerate(flags):
        args.multi_view = flag["multi_view"]
        args.ER = flag["ER"]
        args.EPL =flag["EPL"]
        idx = f"noise_metrics_{model_name[i]}"
        repeat_metrics = []
        repeat_records = []
        for seed in seeds:
            noise_view = noise_views[seed]
            metrics, df, records, results_df = cross_test_noise( args, X, V, y, noise_view, seed)
            repeat_metrics.append(metrics)   # (5,4)
            repeat_records.append(records)
            
        repeat_metrics = np.array(repeat_metrics)  
        # (8, 25, 4) noise_level:7, 1th dim:0 noise
        merged_metrics = repeat_metrics.transpose(1, 0, 2, 3).reshape(8, 25, 4)
        # 保存
        noise_results[idx] = repeat_records
        noise_metrics[idx] = repeat_metrics
        noise_mean_metrics[idx] = merged_metrics.mean(axis=1)              # (8,4)
        noise_std_metrics[idx] = merged_metrics.std(axis=1, ddof=1)        # (8,4)

    return noise_mean_metrics, noise_std_metrics, noise_results, noise_metrics
        
def hyper_parameter(args):
    
    gama = [1,2,4,6,8]
    emb = [1,2,3,4,5]
    param_result = []
    param_metrics = np.zeros([len(gama), len(emb), 4])
    for p in range(len(gama)):
        args.gama = gama[p]
        X, V, y = MV_data('../data/', args.dataset, args.gama, config.cols[args.dataset])
        for q in range(len(emb)):
            args.emb_dim = emb[q]
            metrics, *_ = cross_test( args, X, V, y)
            param_result.append({
            'gama': gama[p],
            'emb_dim': emb[q],
            'metrics':metrics
        })
            param_metrics[p, q, :] = np.mean(metrics, 0)
        
    return param_metrics, param_result





# repeat_metrics, repeat_df, repeat_records = repeat_CV(args, X, V, y)


#############################noise repeat5V
# noise_mean_metrics, noise_std_metrics, noise_results, noise_metrics = noise_result(args, X, V, y)

#############################parameter analysis
# param_metrics, param_result = hyper_parameter(args)

