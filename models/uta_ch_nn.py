# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:44:32 2022

@author: dell
"""

 # -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 19:27:12 2022

@author: dell
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score


class MySigmoid(nn.Module):
    def __init__(self, delta=0.01):
        super(MySigmoid, self).__init__()
        self.delta = delta

    def forward(self, x):
        out = torch.where(x < 0, self.delta * x, 
              torch.where(x > 1, self.delta * (x - 1) + 1, 
              x))
        return out

LeakyHardSigmoid = MySigmoid(delta = 0.01)


def gen_mask(row, L):
    #row 准则数，L隐藏层神经元数
    col = row * L
    mask = np.zeros((row, col))
    for i in range(row):
        mask[i,np.arange(i*L,i*L+L)]=np.ones(L)
    
    return mask

class MyLoss(nn.Module):
    def __init__(self,num_t):
        super(MyLoss, self).__init__()
        self.num_t = num_t
        #阈值
        self.coefficient = nn.Parameter(torch.empty(num_t))
        self.coefficient.data.uniform_(0,1)
        self.coefficient.data = torch.sort(self.coefficient)[0]

        
    def forward(self,s,label):
        label = label.type(torch.long)
        
        t = self.coefficient
        loss = torch.zeros_like(s)
        for i in range(label.shape[0]):
            y = label[i]
            si = s[i]
            if y == 0:
                loss[i] = F.relu(si - t[0])
            elif y == self.num_t:
                loss[i] = F.relu(t[-1] - si)
            else:
                lower = F.relu(t[y - 1] - si)
                upper = F.relu(si - t[y])
                loss[i] = torch.max(lower, upper)
    
        loss = torch.mean(loss)
       
        return loss, t
    
class Mon_block(nn.Module):
    def __init__(self, in_units, L):
        #L单调模块隐藏层神经元数
        super(Mon_block, self).__init__()
        self.in_units = in_units
        self.L = L
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        for i in range(in_units):
            self.layers1.append(nn.Linear(1,L))
            self.layers2.append(nn.Linear(L,1,bias=False))
            
        self.outlayer = nn.Linear(in_units,1,bias=False)
        
    def forward(self, x):
        out1 = []
        for i in range(self.in_units):
            out1.append(LeakyHardSigmoid(self.layers1[i](x[:,i].view(x.shape[0],1))))
        #边际价值
        out2 = []
        for i in range(self.in_units):
            out2.append(self.layers2[i](out1[i]))
        #综合价值
        out = self.outlayer(torch.cat(out2,dim=1))
        return out

    
class UtaModel(nn.Module):
    def __init__(self, in_dim, mon_dim):
        #mon_dim单调模块隐藏层神经元数，att_dim注意力模块隐藏层神经元数
        super(UtaModel, self).__init__()
        self.in_dim = in_dim
        self.mon_dim = mon_dim
        self.Monotone = Mon_block(in_dim, mon_dim)

        
    def forward(self, x):

        mon_out = self.Monotone(x)
        s_out = mon_out
        
        return s_out
    
    
class Ch_posModel(nn.Module):
    def __init__(self, in_dim):
        #mon_dim单调模块隐藏层神经元数，att_dim注意力模块隐藏层神经元数
        super(Ch_posModel, self).__init__()
        self.in_dim = in_dim

        self.layer1 = nn.Linear(in_dim, 1)
        nn.init.constant_(self.layer1.weight, 1/in_dim)
        self.layer2 = nn.Linear(int(in_dim * (in_dim-1) / 2), 1)
        nn.init.constant_(self.layer2.weight, 1/int(in_dim * (in_dim-1) / 2))
    
    def forward(self, x):

        single_out = self.layer1(x)
        num_criteria = x.shape[1]
        row, col = list(), list()
        for i in range(num_criteria - 1):
            for j in range(i + 1, num_criteria):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        bi_interaction = torch.min(p,q)
        int_out = self.layer2(bi_interaction)
        
        out = single_out + int_out
                
        out = torch.sigmoid(out)
        
        return out


