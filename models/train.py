# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:38:21 2024

@author: gyuan
"""
import os
import numpy as np
import pandas as pd
import torch

from mymodel import EPL, MEPL, MEPL_ER, PL, MEPL_scen, Add_model, mse_loss, m_loss
from dataload import one_hot_embedding
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def train(args, num_g, num_t, train_loader, val_loader, fold):
    
    if args.multi_view:
        views = len(num_g)
        if args.ER:
            model = MEPL_ER(num_t, num_g, args.gama, args.emb_dim, views)
        else:
            model = MEPL(num_t, num_g, args.gama, args.emb_dim, views)
        
    else: 
        if args.EPL:
            model = EPL(num_t, num_g, args.gama, args.emb_dim)
        else:
            model = PL(num_t, num_g, args.gama, args.emb_dim)
            CE_loss = torch.nn.CrossEntropyLoss()
        
    num_class = num_t+1
    
    if isinstance(model, MEPL_ER):
        optimizer = torch.optim.AdamW([
            {'params': model.EPL.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
            {'params': model.fusion.parameters(), 'lr': 1e-4, 'weight_decay': args.wd},
            ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', 'acc', 'val_acc'])
    
    for epoch in range(args.num_epoch):
        step = 1
        loss_all = 0
        train_acc_sum = 0
        model.train()
        
        for step, (inputs, labels) in enumerate(train_loader, 1):
                       
            labels = labels.to(device)
            y = one_hot_embedding(labels, num_class)
            y = y.to(device)
            if args.multi_view:
                for v in range(len(inputs)):
                    inputs[v] = inputs[v].to(device)                               
                evidence, evidence_a, alpha, alpha_a = model (inputs)                                
                _, preds = torch.max(evidence_a.data, 1)
                loss = m_loss(y, alpha, alpha_a, epoch, num_class, 10, device = device)
                
            else:
                evidence, _ = model (inputs)
                alpha = evidence + 1
                _, preds = torch.max(evidence, 1)
                if args.EPL:
                    loss = torch.mean(mse_loss(y, alpha, epoch, num_class, 10, device = device)) 
                else:
                    loss = CE_loss(evidence, y)
            
            optimizer.zero_grad()              
            loss.backward()
            optimizer.step()
        
            
            loss_all += loss.item()  
            train_acc = accuracy_score(labels, preds)
            train_acc_sum += train_acc.item()

          
        cur_loss = loss_all / step
        cur_acc = train_acc_sum / step
        
        val_acc, val_precision, val_f1, val_recall, val_alpha  = evaluate(args, model, val_loader, device)
        val_metrics = [val_acc, val_precision, val_f1, val_recall]

        
        info = (epoch, cur_loss, cur_acc, val_acc)
        dfhistory.loc[epoch] = info
        print('Epoch: {:03d}, Loss: {:.4f}, Train_Acc: {:.4f}, Val_Acc: {:.4f}'.
          format(epoch, cur_loss, cur_acc, val_acc))
    
    # model_name = model.__class__.__name__
    # save_path = f"saved_models/{args.dataset}/{model_name}/seed_{args.random_seed}_fold_{fold+1}.pth"
    # torch.save(model.state_dict(), save_path)    
    
    return dfhistory, val_metrics, val_alpha 



def build_model_and_constraints(args, flag, num_t, num_g):
    if len(flag) == 1:
        f = flag[0]
        if f in ["Add", "AddM"]:
            model = Add_model(num_t, num_g, args.gama)
            constrained_layers = [model.linearlayer] if f.endswith("M") else []

        elif f in ["EPL", "EPLM"]:
            model = EPL(num_t, num_g, args.gama, args.emb_dim)
            constrained_layers = [model.FM_layer.linearlayer] if f.endswith("M") else []
        else:
            raise ValueError(f"Unknown flag: {f}")
    else:
        views = len(flag)
        model = MEPL_scen(flag, num_t, num_g, args.gama, args.emb_dim, views)

        CONSTRAINT_PICKERS = {
            "AddM": lambda sub: [sub.linearlayer],
            "EPLM": lambda sub: [sub.FM_layer.linearlayer],}
        constrained_layers = []
        for i, f in enumerate(flag):
            pick = CONSTRAINT_PICKERS.get(f)
            if pick:
                constrained_layers.extend(pick(model.PL[i]))

    return model, constrained_layers

class WeightConstraint:
    def __call__(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            with torch.no_grad():
                module.weight.clamp_(0, 100)   # 或 (0.0, 100.0)，根据你的需求
constraint=WeightConstraint()

def train_scen(args, flag, num_g, num_t, train_loader, val_loader, seed, fold):
    
    model, constrained_layers = build_model_and_constraints(args, flag, num_t, num_g)
    num_class = num_t+1
    if len(flag)>1:
        optimizer = torch.optim.AdamW([
            {'params': model.PL.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
            {'params': model.fusion.parameters(), 'lr': 1e-4, 'weight_decay': args.wd},
            ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', 'acc', 'val_acc'])
    
    for epoch in range(args.num_epoch):
        step = 1
        loss_all = 0
        train_acc_sum = 0
        model.train()
        
        for step, (inputs, labels) in enumerate(train_loader, 1):
                       
            labels = labels.to(device)
            y = one_hot_embedding(labels, num_class)
            y = y.to(device)
            if args.multi_view:
                for v in range(len(inputs)):
                    inputs[v] = inputs[v].to(device)                               
                evidence, evidence_a, alpha, alpha_a = model (inputs)                                
                _, preds = torch.max(evidence_a.data, 1)
                loss = m_loss(y, alpha, alpha_a, epoch, num_class, 10, device = device)
            else:
                evidence, _ = model (inputs)
                alpha = evidence + 1
                _, preds = torch.max(evidence, 1)
                loss = torch.mean(mse_loss(y, alpha, epoch, num_class, 10, device = device)) 

                
            optimizer.zero_grad()              
            loss.backward()
            optimizer.step()
            for layer in constrained_layers:
                constraint(layer)
            
            loss_all += loss.item()  
            train_acc = accuracy_score(labels, preds)
            train_acc_sum += train_acc.item()
          
        cur_loss = loss_all / step
        cur_acc = train_acc_sum / step
        
        val_acc, val_precision, val_f1, val_recall, val_alpha  = evaluate(args, model, val_loader, device)
        val_metrics = [val_acc, val_precision, val_f1, val_recall]
        
        info = (epoch, cur_loss, cur_acc, val_acc)
        dfhistory.loc[epoch] = info
        print('Epoch: {:03d}, Loss: {:.4f}, Train_Acc: {:.4f}, Val_Acc: {:.4f}'.
          format(epoch, cur_loss, cur_acc, val_acc))
    

    # model_name = model.__class__.__name__
    # flag_name = "_".join(flag)   
    # save_dir = os.path.join("saved_models", "benchmark", args.dataset, flag_name)
    # os.makedirs(save_dir, exist_ok=True)
   
    # save_path = os.path.join(save_dir, f"{model_name}_seed_{seed}_fold_{fold+1}.pth")
    # torch.save(model.state_dict(), save_path)
 
    
    return dfhistory, val_metrics, val_alpha

def evaluate(args, model, dataloader, device):
    
    model.eval()

    y_pred = []
    y_true = []
    out_alpha = []
    out_alpha_view = []
    val_loss_sum = 0
    step = 1
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(dataloader):

            if args.multi_view:
                evidence, evidence_a, alpha, alpha_a = model (inputs)
                _, preds = torch.max(evidence_a.data, 1)
                out_alpha.append(alpha_a)
                out_alpha_view.append({k: v.detach().cpu().numpy() for k, v in alpha.items()})
            else:
                evidence,_ = model (inputs)
                _, preds = torch.max(evidence, 1)
                out_alpha.append(evidence)
            y_pred.append(preds)
            y_true.append(labels.data)
            

    
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
   
    out_alpha = torch.cat(out_alpha)
    # out_alpha_merge = {
    #     k: np.concatenate([d[k] for d in out_alpha_view], axis=0)
    #     for k in out_alpha_view[0]
    # }
    # out_alpha_merge = np.concatenate([out_alpha_merge[i] for i in range(len(out_alpha_merge))], axis=1)
    # records = pd.DataFrame({
    # 'pred': y_pred.detach().cpu().numpy(),
    # 'true': y_true.detach().cpu().numpy(),
    # 'a': list(out_alpha.detach().cpu().numpy()),
    # 'a_view': list(out_alpha_merge)
    # })
    
    records = pd.DataFrame({
    'pred': y_pred.detach().cpu().numpy(),
    'true': y_true.detach().cpu().numpy(),
    'a': list(out_alpha.detach().cpu().numpy()),})
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    return acc, precision, f1, recall, records




def test_noise(args, num_g, num_t, dataloader, fold, seed):
    
    if args.multi_view:
        views = len(num_g)
        if args.ER:
            model = MEPL_ER(num_t, num_g, args.gama, args.emb_dim, views)
        else:
            model = MEPL(num_t, num_g, args.gama, args.emb_dim, views)
        
    else: 
        if args.EPL:
            model = EPL(num_t, num_g, args.gama, args.emb_dim)
        else:
            model = PL(num_t, num_g, args.gama, args.emb_dim)
            
    model_name = model.__class__.__name__
    save_path = f"saved_models/save_0325/{args.dataset}/{model_name}/seed_{seed}_fold_{fold+1}.pth"
    model.load_state_dict(torch.load(save_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_acc, val_precision, val_f1, val_recall, records  = evaluate(args, model, dataloader, device)
    val_metrics = [val_acc, val_precision, val_f1, val_recall]
    
    return val_metrics, records

