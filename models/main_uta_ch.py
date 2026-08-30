# -*- coding: utf-8 -*-
"""

"""
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from uta_ch_nn import MyLoss, UtaModel, Ch_posModel
from metrics import metric_measures

torch.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HCDR', help='which dataset to use')
parser.add_argument('--modelname', type=str, default='ch', help='which model to test')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weightdecay')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('--hidden_layer', type=int, default= 20, help='neural hidden layer')
parser.add_argument('--random_seed', type=int, default=2024, help='size of common item be counted')
# parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
args = parser.parse_args()


 
class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            w=w.clamp(0,100) #将参数范围限制到0-100之间
            module.weight.data=w
        elif hasattr(module,'coefficient'):
            p=module.coefficient.data
            p=p.clamp(0,1) #将参数范围限制到0-1之间
            module.coefficient.data=p

constraints=weightConstraint()

def train(args, data_info, fold):
    train_loader = data_info[0]
    val_loader = data_info[1]
    in_dim = data_info[2]
    t_dim = data_info[3]
    mon_dim = args.hidden_layer
    max_acc = 0
    min_loss = 1
    
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', 'acc', 'val_acc'])
    if args.modelname == 'uta':
        model= UtaModel(in_dim, mon_dim)
        loss_func = MyLoss(t_dim)
    elif args.modelname == 'ch':
        model= Ch_posModel(in_dim)
        loss_func = MyLoss(t_dim)
    

    optimizer = torch.optim.AdamW([{"params":model.parameters()},{"params":loss_func.parameters()}], lr=args.lr, weight_decay= args.weightdecay)


    print([i.size() for i in filter(lambda p: p.requires_grad, model.parameters())])
    print('start training...')
    for epoch in range(args.n_epoch):
        # training
        step = 1
        loss_all = 0
        train_acc_sum = 0
        model.train()
        
        for step, (features, labels) in enumerate(train_loader, 1):
           
           
            output = model(features)
            loss, t = loss_func(output, labels)
            loss_all += loss.item()

            train_acc = metric_measures(output, labels, t)[0]
            
            train_acc_sum += train_acc.item()
            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()
            

        cur_loss = loss_all / step
        cur_acc = train_acc_sum / step


        val_acc, val_precision, val_f1, val_recall, records = evaluate(model, loss_func, val_loader) 
        
        info = (epoch, cur_loss, cur_acc, val_acc)
        dfhistory.loc[epoch-1] = info
        val_metrics = [val_acc, val_precision, val_f1, val_recall]
        print('Epoch: {:03d}, Loss: {:.4f}, Train_Acc: {:.4f}, Val_Acc: {:.4f},'.
          format(epoch, cur_loss, cur_acc, val_acc))
        
            
    return dfhistory, val_metrics, records

def evaluate(model, loss_func, loader):
    
    model.eval()

    predictions = []
    labels = []
    val_loss_sum = 0
    val_step = 1
    with torch.no_grad():
        for val_step, (feature, label) in enumerate(loader, 1):
            pred = model(feature)
            predictions.append(pred)
            labels.append(label)
            val_loss, t = loss_func(pred, label)
            val_loss_sum += val_loss.item()
    
    y_pred = torch.cat(predictions)
    y_true = torch.cat(labels)
    
    acc, precision, f1, recall, preds = metric_measures(y_pred, y_true, t)
    loss = val_loss_sum / val_step
    records = pd.DataFrame({
    'pred': preds.detach().cpu().numpy(),
    'true': y_true.detach().cpu().numpy()})
    
    return acc, precision, f1, recall, records

##交叉验证
def cross_val( args, X, y, seed):
    
    cv =5
    t_dim = max(y)
    skf = StratifiedKFold(n_splits=cv, shuffle = True, random_state= seed)      
    metrics = np.zeros((5,4))
    df = pd.DataFrame()
    records = pd.DataFrame()
    
    for i, (train_idx, test_idx) in enumerate(skf.split(X[0], y)):
        
        X_train, y_train = [arr[train_idx] for arr in X], y[train_idx] 
        X_test, y_test = [arr[test_idx] for arr in X], y[test_idx]
        
        X_train,X_test=np.concatenate(X_train, axis=1),np.concatenate(X_test, axis=1)
        in_dim = X_train.shape[1]
        
        
        train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
        test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle = True)

        datainfo = [train_loader, test_loader, in_dim, t_dim, test_loader]
        df_fold, test_metrics, test_records = train(args, datainfo, fold = i)

        metrics[i,:] = test_metrics
        records = pd.concat([records, test_records])
        df = pd.concat([df, df_fold])
    
    return metrics, records, df




def repeat_CV(args, X, y):
    seeds = [0,1,2,3,4]
    repeat_metrics = []
    repeat_df = []
    repeat_records = []
    for seed in seeds:       
        metrics, records, df = cross_val( args, X, y, seed)
        
        repeat_metrics.append(metrics)
        repeat_df.append(df)
        repeat_records.append(records)
    
    return repeat_metrics, repeat_df, repeat_records
        

with open(f"../data/{args.dataset}/{args.dataset}4.pkl", "rb") as f:
    data = pickle.load(f)
    X = data['X']
    V = data["V"]
    y = data["y"]

# uta_repeat_metrics, uta_repeat_df, uta_repeat_records = repeat_CV(args, X, y) 
ch_repeat_metrics, ch_repeat_df, ch_repeat_records = repeat_CV(args, X, y) 
