# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:26:28 2024

@author: gyuan
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from dataload import one_hot_embedding

class FM_layer(nn.Module):     
    def __init__(self, num_g, gama, emb_dim):
        super().__init__()
        self.num_g = num_g
        self.gama = gama
        self.emb_dim = emb_dim
        self.linearlayer = nn.Linear(num_g*gama, 1)
        self.V = nn.Parameter(torch.empty(num_g*gama, emb_dim))
        nn.init.xavier_normal_(self.V)
        

    def forward(self, X):
        add_out = self.linearlayer(X)
        # u = self.linearlayer.weight.data
        X = X.unsqueeze(2)
        A = self.V * X         
        pow_of_sum = torch.sum(A, dim=1) ** 2 # -> [batch_size, emb]
        sum_of_pow = torch.sum(A ** 2, dim=1) # -> [batch_size, emb]
        # -> [batch_size]
        fm_out = torch.sum(pow_of_sum - sum_of_pow, dim=1, keepdim=True) * 0.5
        S_out = add_out + fm_out
        
        return S_out
     
class sort_layer(nn.Module):     
    def __init__(self, num_t):
        super().__init__()
        self.num_t = num_t
        self.t = nn.Parameter(torch.randn(num_t))
        self.t.data = torch.sort(self.t)[0]
        

    def forward(self, s):

        sigma_1 = self.t[0] - s
        sigma_H = s-self.t[-1]
        sigma_h_1 = s - self.t[:-1]
        sigma_h_2 = self.t[1:] - s
        sigma_h = torch.min(sigma_h_1, sigma_h_2)
        
        ci = torch.cat((sigma_1,sigma_h,sigma_H),dim=1)
               
        return ci
    
class Add_model(nn.Module):     
    def __init__(self, num_t, num_g, gama ):
        super().__init__()
        self.num_g = num_g
        self.gama = gama
        self.linearlayer = nn.Linear(num_g*gama, 1)
        self.sort_layer = sort_layer(num_t)
        self.num_t = num_t
        
    def forward(self, X):
        S_out = self.linearlayer(X)
        prob = self.sort_layer(S_out)
        # evidence = F.softmax(prob, dim = 1)
        evidence = F.softplus(prob, 1)  
        
        return evidence, prob
    
    
class PL(nn.Module):
         
    def __init__(self, num_t, num_g, gama, emb_dim):
        super().__init__()
        self.FM_layer = FM_layer(num_g, gama, emb_dim)
        self.sort_layer = sort_layer(num_t)
        self.num_t = num_t
         
 
    def forward(self, X):
        S_out = self.FM_layer(X)
        evidence = self.sort_layer(S_out)      
        
        return evidence, S_out
     
  
class EPL(nn.Module):
         
    def __init__(self, num_t, num_g, gama, emb_dim):
        super().__init__()
        self.FM_layer = FM_layer(num_g, gama, emb_dim)
        self.sort_layer = sort_layer(num_t)
        self.num_t = num_t
         
 
    def forward(self, X):
        S_out = self.FM_layer(X)
        evidence = F.softplus(self.sort_layer(S_out), 1)
        
        alpha = evidence + 1
               
        return evidence, S_out

class MEPL(nn.Module):     
    def __init__(self, num_t, dims, gama, emb_dim, views):
        super().__init__()
        self.num_t = num_t
        self.views = views
        self.dims = dims
        self.EPL = nn.ModuleList([EPL(self.num_t, dims[i], gama, emb_dim) for i in range(self.views)])
        
    def DS_Combin(self, alpha):
       """
       :param alpha: All Dirichlet distribution parameters.
       :return: Combined Dirichlet distribution parameters.
       """
       def DS_Combin_two(alpha1, alpha2):
           """
           :param alpha1: Dirichlet distribution parameters of view 1
           :param alpha2: Dirichlet distribution parameters of view 2
           :return: Combined Dirichlet distribution parameters
           """
           num_class = self.num_t + 1
           alpha = dict()
           alpha[0], alpha[1] = alpha1, alpha2
           b, S, E, u = dict(), dict(), dict(), dict()
           for v in range(2):
               S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
               E[v] = alpha[v]-1
               b[v] = E[v]/(S[v].expand(E[v].shape))
               u[v] = num_class/S[v]

           # b^0 @ b^(0+1)
           bb = torch.bmm(b[0].view(-1, num_class, 1), b[1].view(-1, 1, num_class))
           # b^0 * u^1
           uv1_expand = u[1].expand(b[0].shape)
           bu = torch.mul(b[0], uv1_expand)
           # b^1 * u^0
           uv_expand = u[0].expand(b[0].shape)
           ub = torch.mul(b[1], uv_expand)
           # calculate C
           bb_sum = torch.sum(bb, dim=(1, 2), out=None)
           bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
           C = bb_sum - bb_diag

           # calculate b^a
           b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
           # calculate u^a
           u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

           # calculate new S
           S_a = num_class / u_a
           # calculate new e_k
           e_a = torch.mul(b_a, S_a.expand(b_a.shape))
           alpha_a = e_a + 1
           return alpha_a

       for v in range(len(alpha)-1):
           if v==0:
               alpha_a = DS_Combin_two(alpha[0], alpha[1])
           else:
               alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
       return alpha_a
       
    def forward(self, X):
          evidence = self.infer(X)
          alpha = dict()
          for v_num in range(len(X)):
              alpha[v_num] = evidence[v_num] + 1
              
          alpha_a = self.DS_Combin(alpha)
          evidence_a = alpha_a - 1
          
          return evidence, evidence_a, alpha, alpha_a
    
    def infer(self, input):

          evidence = dict()
          for v_num in range(self.views):
              evidence[v_num],_ = self.EPL[v_num](input[v_num])
          return evidence
    



class fusion_layer(nn.Module):
         
    def __init__(self, views, classes):
        super().__init__()
        self.views = views
        self.classes = classes
        self.w = nn.Parameter(torch.ones(views), requires_grad=True)
            
    def ERA_Combin(self, b, u, u_bar):
        
        
        bb_bu_ub = torch.prod(b + u_bar + u, dim=0)
        uu = torch.prod(u_bar + u, dim=0)    
        k = (torch.sum(bb_bu_ub, dim = 1, keepdim = True) - (self.classes -1) * uu) ** -1
        
        b_a = (bb_bu_ub - uu) * k
        u_a = (uu - torch.prod(u_bar, dim = 0)) * k
        u_bar_a = torch.prod(u_bar, dim = 0) * k
        
        b_a = b_a / (1-u_bar_a)
        u_a = u_a / (1-u_bar_a)     
        # calculate new S
        S_a = self.classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1 
        
        return alpha_a
     
    def forward(self, alpha):
        w = nn.functional.softmax(self.w, dim=0)
        b, S, E, u, u_bar = dict(), dict(), dict(), dict(), dict()
        num_class = self.classes       
        for v in range(self.views):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = num_class/S[v]
        b = torch.stack(list(b.values()), dim = 0)
        u = torch.stack(list(u.values()), dim = 0)
        b = b * w.view(self.views, 1, 1)
        u = u * w.view(self.views, 1, 1)
        u_bar = 1- w.view(self.views, 1, 1)
        alpha_a = self.ERA_Combin(b, u, u_bar)
        
        return alpha_a
         
class MEPL_ER(nn.Module):     
    def __init__(self, num_t, dims, gama, emb_dim, views):
        super().__init__()
        self.num_t = num_t
        self.views = views
        self.dims = dims
        self.EPL = nn.ModuleList([EPL(self.num_t, dims[i], gama, emb_dim) for i in range(self.views)])
        self.fusion = fusion_layer(views, num_t+1)
        
       
    def forward(self, X):
          evidence = self.infer(X)
          alpha = dict()
          for v_num in range(len(X)):
              alpha[v_num] = evidence[v_num] + 1
              
          alpha_a = self.fusion(alpha)
          evidence_a = alpha_a - 1
          
          return evidence, evidence_a, alpha, alpha_a
    
    def infer(self, inputs):

          evidence = dict()
          for v_num in range(self.views):
              evidence[v_num],_ = self.EPL[v_num](inputs[v_num])
          return evidence         


class weighted_mean_fusion_layer(nn.Module):

    def __init__(self, views):
        super().__init__()
        self.views = views
        self.w = nn.Parameter(torch.ones(views), requires_grad=True)

    def forward(self, alpha):
        w = nn.functional.softmax(self.w, dim=0)
        prob = []

        for v in range(self.views):
            # Dirichlet strength
            S_v = torch.sum(alpha[v], dim=1, keepdim=True)
            # Expected class probability under Dirichlet distribution
            p_v = alpha[v] / S_v
            prob.append(p_v)

        # shape: [views, batch_size, classes]
        prob = torch.stack(prob, dim=0)
        # weighted mean class probabilities
        prob_a = torch.sum(prob * w.view(self.views, 1, 1),dim=0)

        return prob_a

class MEPL_WM(nn.Module):

    def __init__(self, num_t, dims, gama, emb_dim, views):
        super().__init__()

        self.num_t = num_t
        self.dims = dims ###number of criteria in each view
        self.views = views

        self.EPL = nn.ModuleList([EPL(self.num_t, dims[i], gama, emb_dim)for i in range(self.views)])

        self.fusion = weighted_mean_fusion_layer(views)

    def forward(self, X):

        evidence = self.infer(X)

        alpha = dict()

        for v_num in range(len(X)):
            alpha[v_num] = evidence[v_num] + 1

        # weighted mean class probabilities
        prob_a = self.fusion(alpha)

        return evidence, alpha, prob_a

    def infer(self, inputs):

        evidence = dict()

        for v_num in range(self.views):
            evidence[v_num], _ = self.EPL[v_num](inputs[v_num])

        return evidence

class MEPL_scen(nn.Module):     
    def __init__(self, flags, num_t, dims, gama, emb_dim, views):
        super().__init__()
        self.num_t = num_t
        self.views = views
        self.dims = dims
        # REG = {
        #     "Add": lambda dim: Add_model(num_t=self.num_t, num_g =dim, gama=gama),
        #     "Int": lambda dim: EPL(num_t=self.num_t, num_g =dim, gama=gama, emb_dim = emb_dim),
        #     }
        REG = {
        "Add":  lambda dim: Add_model(num_t=self.num_t, num_g=dim, gama=gama),
        "EPL":  lambda dim: EPL(num_t=self.num_t, num_g=dim, gama=gama, emb_dim=emb_dim),
        "AddM": lambda dim: Add_model(num_t=self.num_t, num_g=dim, gama=gama),
        "EPLM": lambda dim: EPL(num_t=self.num_t, num_g=dim, gama=gama, emb_dim=emb_dim),}
       
        self.PL = nn.ModuleList(
            REG[f](dims[i]) for i, f in enumerate(flags))
        self.fusion = fusion_layer(views, num_t+1)
        
       
    def forward(self, X):
          evidence = self.infer(X)
          alpha = dict()
          for v_num in range(len(X)):
              alpha[v_num] = evidence[v_num] + 1
          alpha_a = self.fusion(alpha)
          evidence_a = alpha_a - 1
          return evidence, evidence_a, alpha, alpha_a
    
    def infer(self, inputs):

          evidence = dict()
          for v_num in range(self.views):             
              evidence[v_num], _ = self.PL[v_num](inputs[v_num])
              
          return evidence                 
           
def kl_divergence(alpha, c, device =None):
    beta = torch.ones((1, c), device = device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    # kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB
    return kl
        

def loglikelihood_loss(y, alpha, device= None):

    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device = None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device = device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device = device)
    # loss = loglikelihood + kl_div
    return loglikelihood + kl_div


def m_loss(y, alpha, alpha_a, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y.to(device)
    loss = 0
    
    loss += mse_loss(y, alpha_a, epoch_num, num_classes, annealing_step, device)
    loss = torch.mean(loss)

    return loss


