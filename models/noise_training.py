# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:20:48 2026

@author: dell
"""
import os
import config
import numpy as np
import pandas as pd
import pickle
import torch
from dataload import load_data, preprocess_data, preprocess_data_fast, MV_data, MultiViewDataset
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader, Subset
from train import train
from itertools import permutations

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
parser.add_argument('--ER', type=bool, default=True, help='if ER fusion')
parser.add_argument('--EPL', type=bool, default=False, help='if uncertain estimation')
args = parser.parse_args()


with open("../data/BCW/BCW" + str(args.gama) +".pkl", "rb") as f:
    data = pickle.load(f)
    X = data['X']
    V = data["V"]
    y = data["y"]
# X, V, y = MV_data('../data/', args.dataset, args.gama, config.cols[args.dataset])
num_t = max(y)##label from 0


def generate_noise(data, mean, std, seed=args.random_seed):
    np.random.seed(seed)  # 设置随机种子
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def cross_train_noise(args, X, V, y, seed):

    cv = 5
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    metrics = np.zeros([cv, 4])
    df = pd.DataFrame()
    records = pd.DataFrame()

    num_t = max(y)

    for i, (train_idx, test_idx) in enumerate(skf.split(X[0], y)):

        X_train, V_train, y_train = [arr[train_idx] for arr in X], [arr[train_idx] for arr in V], y[train_idx] 
        X_test, V_test, y_test = [arr[test_idx] for arr in X], [arr[test_idx] for arr in V], y[test_idx]

        num_g = [X[v].shape[1] for v in range(len(X))]
        train_dataset = MultiViewDataset(V_train, y_train)
        test_dataset = MultiViewDataset(V_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle= True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle= True)

        df_fold, test_metrics, test_records = train(args, num_g, num_t, train_loader, test_loader, fold=i)

        metrics[i, :] = test_metrics
        records = pd.concat([records, test_records])
        df = pd.concat([df, df_fold])

    return metrics, df, records

def repeat_CV_noise(args, X, V, y, noise_view, noise_std=0.1):
    
    seeds = [0,1,2,3,4]
    repeat_metrics = []
    repeat_df = []
    repeat_records = []
    for seed in seeds:
        args.random_seed = seed
        X_noise = [arr.copy() for arr in X]
        V_noise = [arr.copy() for arr in V]
        X_noise[noise_view] = generate_noise(X[noise_view], mean=0, std=noise_std, seed=seed)
        x_min, x_max = X[noise_view].min(axis=0), X[noise_view].max(axis=0)
        V_noise[noise_view] = preprocess_data_fast(X_noise[noise_view], args.gama, x_min, x_max)
        
        metrics, df, records = cross_train_noise( args, X_noise, V_noise, y, seed)
        repeat_metrics.append(metrics)
        repeat_df.append(df)
        repeat_records.append(records)
    
    return repeat_metrics, repeat_df, repeat_records


def load_noise_weights(args, save_tag, seeds=[0, 1, 2, 3, 4], n_folds=5):

    all_weights = []

    for seed in seeds:
        for fold in range(n_folds):

            model_path = (
                f"saved_models/noise_train/"
                f"{args.dataset}/"
                f"{save_tag}/"
                f"seed_{seed}_fold_{fold+1}.pth"
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError( f"Model not found: {model_path}")

            # load state_dict
            state_dict = torch.load( model_path, map_location="cpu")
            # fusion.w is the parameter before softmax
            raw_w = state_dict["fusion.w"]
            # normalized view weights
            w = torch.softmax(raw_w, dim=0)

            all_weights.append(w.detach().cpu().numpy())

    all_weights = np.array(all_weights)

    weight_mean = all_weights.mean(axis=0)
    weight_std = all_weights.std(axis=0, ddof=1)

    return all_weights, weight_mean, weight_std


noise_levels = [0.01,0.1,1]

# all_noise_results = {}
# save_dir = '../testdata/repeatCV/noise_trian'
# os.makedirs(save_dir, exist_ok=True)


# for noise_std in noise_levels:

#     for noise_view in range(len(X)):

#         print(f"Noise level={noise_std}, view={noise_view+1}")

#         args.save_tag = ( f"sigma_{noise_std}_view_{noise_view+1}")

#         repeat_metrics, repeat_df, repeat_records = repeat_CV_noise(args, X, V, y, noise_view=noise_view, noise_std=noise_std )
#         mean_metric = np.vstack(repeat_metrics).mean(axis=0)
#         std_metric = np.vstack(repeat_metrics).std(axis=0, ddof=1)
#         key = (  f"sigma_{noise_std}_view_{noise_view+1}" )

#         all_noise_results[key] = {
#             "noise_std": noise_std,
#             "noise_view": noise_view,
#             "metrics": repeat_metrics,
#             "df": repeat_df,
#             "records": repeat_records,
#             "mean": mean_metric,
#             "std": std_metric
#         }
#         all_results_path = os.path.join(save_dir, 'all_results.pkl')
#         with open(all_results_path, 'wb') as f:
#             pickle.dump(all_noise_results, f)

###读取权重
all_weight_results = {}

for noise_std in [0.01, 0.1, 1]:

    for noise_view in range(3):

        save_tag = ( f"sigma_{noise_std}_view_{noise_view+1}" )

        weights, weight_mean, weight_std = load_noise_weights(args, save_tag )

        all_weight_results[save_tag] = {
            "weights": weights,
            "mean": weight_mean,
            "std": weight_std
        }
        
def generate_weight_latex(all_weight_results, decimals=3):

    rows = []

    for key, result in all_weight_results.items():

        # parse key
        # sigma_0.01_view_1
        parts = key.split('_')
        sigma = parts[1]
        view = parts[3]

        mean = result["mean"]
        std = result["std"]

        weights = [
            f"{mean[i]:.{decimals}f}$\\pm${std[i]:.{decimals}f}"
            for i in range(len(mean))
        ]

        rows.append([
            sigma,
            f"View {view}",
            *weights
        ])

    latex = ""

    for row in rows:
        latex += (
            f"{row[0]} & {row[1]} & "
            f"{row[2]} & {row[3]} & {row[4]} \\\\\n"
        )

    return latex

latex_rows = generate_weight_latex( all_weight_results, decimals=3)