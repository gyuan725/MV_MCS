# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 20:22:55 2023

@author: dell
"""

import pandas as pd
import numpy as np
import torch


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            


def metric_measures(s_pred, y_label, t):
    
    
    t = torch.cat([torch.tensor([float('-inf')]),t])
    t = torch.cat([t,torch.tensor([float('inf')])])
    y_pred = torch.zeros(y_label.shape[0]) - 1
    for i in range(y_label.shape[0]):
        for j in range(t.shape[0]-1):          
            if t[j] <= s_pred[i] < t[j+1]:
                y_pred[i] = j

    acc = accuracy_score(y_label, y_pred)
    precision = precision_score(y_label, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_label, y_pred, average='macro')
    recall = recall_score(y_label, y_pred, average='macro', zero_division=0)

    return acc, precision, f1, recall, y_pred

