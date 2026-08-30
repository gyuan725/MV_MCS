# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 20:05:54 2026

@author: dell
"""

import os
import numpy as np
import pandas as pd
import torch
from mymodel import MEPL_WM
from dataload import one_hot_embedding
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import config
import pickle
from dataload import MV_data, MultiViewDataset
import argparse
from sklearn.model_selection import  StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BCW', help='which dataset to use')
parser.add_argument('--gama', type=int, default=4, help='number of piecewise ponit')
parser.add_argument('--emb_dim', type=int, default=2, help='dimension of latent vector')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--num_epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('--random_seed', type=int, default=2024, help='random_seed')
parser.add_argument('--multi_view', type=bool, default= True, help='if multi_view')
args = parser.parse_args()


# file_path = f"../data/{args.dataset}/{args.dataset}{args.gama}.pkl"
# with open(file_path, "rb") as f:
#     data = pickle.load(f)
#     X = data['X']
#     V = data["V"]
#     y = data["y"]
# # X, V, y = MV_data('../data/', args.dataset, args.gama, config.cols[args.dataset])
# num_t = max(y)##label from 0

def train_WM(args, num_g, num_t, train_loader, val_loader, fold):

    views = len(num_g)
    model = MEPL_WM(num_t,num_g,args.gama,args.emb_dim,views)
        
    num_class = num_t+1

    optimizer = torch.optim.AdamW([
            {'params': model.EPL.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
            {'params': model.fusion.parameters(), 'lr': 1e-4, 'weight_decay': args.wd},
            ])
      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', 'acc', 'val_acc'])
    
    for epoch in range(args.num_epoch):
        step = 1
        loss_all = 0
        train_acc_sum = 0
        model.train()
        
        for step, (inputs, labels) in enumerate(train_loader, 1):
                       
            labels = labels.long().to(device)
            y = one_hot_embedding(labels, num_class)
            y = y.to(device)
            
            for v in range(len(inputs)):
                inputs[v] = inputs[v].to(device)                               
            evidence, alpha, prob_a = model (inputs)                                
            _, preds = torch.max(prob_a.data, 1)
            loss = torch.mean( torch.sum((y - prob_a) ** 2, dim=1) )
        
            optimizer.zero_grad()              
            loss.backward()
            optimizer.step()
            
            loss_all += loss.item()  
            train_acc = accuracy_score(labels, preds)
            train_acc_sum += train_acc.item()
          
        cur_loss = loss_all / step
        cur_acc = train_acc_sum / step
        
        val_acc, val_precision, val_f1, val_recall, val_alpha  = evaluate_WM(args, model, val_loader, device)
        val_metrics = [val_acc, val_precision, val_f1, val_recall]
        
        info = (epoch, cur_loss, cur_acc, val_acc)
        dfhistory.loc[epoch] = info
        print('Epoch: {:03d}, Loss: {:.4f}, Train_Acc: {:.4f}, Val_Acc: {:.4f}'.
          format(epoch, cur_loss, cur_acc, val_acc))
    
    # model_name = model.__class__.__name__
    # save_path = f"saved_models/weighted_mean/{args.dataset}/{model_name}/seed_{args.random_seed}_fold_{fold+1}.pth"
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # torch.save(model.state_dict(), save_path)     
    
    return dfhistory, val_metrics, val_alpha

def evaluate_WM(args, model, dataloader, device):
    
    model.eval()
      
    y_pred = []
    y_true = []
    out_prob = []
    val_loss_sum = 0
    step = 1
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(dataloader):

            evidence, alpha, prob_a = model (inputs)
            _, preds = torch.max(prob_a.data, 1)
            out_prob.append(prob_a)
                
            y_pred.append(preds)
            y_true.append(labels.data)
    
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
   
    out_prob = torch.cat(out_prob)
    
    records = pd.DataFrame({
    'pred': y_pred.detach().cpu().numpy(),
    'true': y_true.detach().cpu().numpy(),
    'prob': list(out_prob.detach().cpu().numpy())})
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    return acc, precision, f1, recall, records

def cross_test( args, X, V, y, seed):
    
    cv=5
    skf = StratifiedKFold(n_splits=cv, shuffle= True, random_state= seed) 
    metrics = np.zeros([cv,4])
    df = pd.DataFrame()
    records = pd.DataFrame()
    num_t = max(y)
    
    for i, (train_idx, test_idx) in enumerate(skf.split(X[0], y)):
        
        X_train, V_train, y_train = [arr[train_idx] for arr in X], [arr[train_idx] for arr in V], y[train_idx] 
        X_test, V_test, y_test = [arr[test_idx] for arr in X], [arr[test_idx] for arr in V], y[test_idx]
        
        
        num_g = [X[i].shape[1] for i in range(len(X))]
        train_dataset = MultiViewDataset(V_train, y_train)
        test_dataset = MultiViewDataset(V_test, y_test)
            
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle= True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle= True)
        df_fold, test_metrics, test_records = train_WM(args, num_g, num_t, train_loader, test_loader, fold = i)
        metrics[i,:] = test_metrics
        records = pd.concat([records, test_records])
        df = pd.concat([df, df_fold]) 
        
    return metrics, df, records  

def repeat_CV(args, X, V, y):
    
    seeds = [0,1,2,3,4]
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

batch_size_map = {
    'BCW': 16,
    'AD': 32,
    'CCA': 64,
    'HCDR': 256,}
emb_dim_map = {
    'BCW': 2,
    'AD': 2,
    'CCA': 4,
    'HCDR': 4,}
epoch_map ={'AD': 200,}
all_results = {}

default_batch_size = args.batch_size
default_num_epoch = args.num_epoch
default_emb_dim = args.emb_dim

save_dir = '../testdata/repeatCV/weighted_mean'
os.makedirs(save_dir, exist_ok=True)

for ds in ['BCW', 'AD', 'CCA', 'HCDR']:
    
    torch.manual_seed(2024)
    np.random.seed(2024)

    args.dataset = ds
    args.batch_size = batch_size_map.get(ds, default_batch_size)
    args.emb_dim = emb_dim_map.get(ds, default_emb_dim)
    args.num_epoch = epoch_map.get(ds, default_num_epoch)
    
    file_path = f"../data/{args.dataset}/{args.dataset}{args.gama}.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        X = data['X']
        V = data["V"]
        y = data["y"]
    num_t = max(y)##label from 0

    repeat_metrics, repeat_df, repeat_records = repeat_CV( args, X, V, y)
    mean_metric = np.vstack(repeat_metrics).mean(axis=0)
    std_metric = np.vstack(repeat_metrics).std(axis=0, ddof=1)
    dataset_results = {
        'metrics': repeat_metrics,
        'df': repeat_df,
        'records': repeat_records,
        'mean': mean_metric,
        'std': std_metric
    }
    dataset_path = os.path.join(save_dir, f'{ds}_results.pkl')
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset_results, f)
    all_results[ds] = dataset_results
    all_results_path = os.path.join(save_dir, 'all_results.pkl')
    with open(all_results_path, 'wb') as f:
        pickle.dump(all_results, f)

# repeat_metrics, repeat_df, repeat_records = repeat_CV( args, X, V, y)