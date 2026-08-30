# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:18:52 2026

@author: gyuan
"""

import os
import joblib
import config
import numpy as np
import pandas as pd
import pickle
import torch
from dataload import load_data,preprocess_data, multiview_data, MV_data, MultiViewDataset, MultiViewDataset_id,preprocess_data_fast
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from torch.utils.data import TensorDataset, DataLoader, Subset
from train import train, train_ER, test_noise, show_case, train_pm, train_scen, train_case
from scipy.stats import wilcoxon
from itertools import combinations


torch.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='DBS', help='which dataset to use')
parser.add_argument('--gama', type=int, default=4, help='number of piecewise ponit')
parser.add_argument('--emb_dim', type=int, default=2, help='dimension of latent vector')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--batch_size', type=int, default = 16, help='batch size')
parser.add_argument('--num_epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('--random_seed', type=int, default=2024, help='random_seed')
parser.add_argument('--multi_view', type=bool, default= False, help='if multi_view')
parser.add_argument('--ER', type=bool, default= True, help='if ER fusion')
parser.add_argument('--EPL', type=bool, default = False, help='if uncertain estimation')
parser.add_argument('--PM', type=str, default = 'Int', help='if uncertain estimation')
args = parser.parse_args()

def MV_benchmark(dataset, gama):
    path = f"../data/benchmark/{dataset}/{dataset}.csv"
    df = pd.read_csv(path)
    y = df['Class'].values.astype(int)
    if min(y) == 1:
        y = y - 1
    df = df.drop(columns = ['Class'])
    X = df.values
    V = preprocess_data(X, gama)

    data = {"X": X, "V": V, "y": y}
    with open(path + str(gama) , "wb") as f:
        pickle.dump(data, f)

    return X, V, y




def cross_test( args, X, V, y, seed, flag):
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
            num_g = X[0].shape[1]
            train_dataset = TensorDataset(torch.tensor(V_train[0]).float(), torch.tensor(y_train).float())
            test_dataset = TensorDataset(torch.tensor(V_test[0]).float(), torch.tensor(y_test).float())
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle= True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle= True)
        df_fold, test_metrics, test_records = train_scen(args, flag, num_g, num_t, train_loader, test_loader, seed, fold = i)
        metrics[i,:] = test_metrics
        records = pd.concat([records, test_records])
        df = pd.concat([df, df_fold])
            
    return metrics, df, records  

def repeat_CV(args, X, V, y, flag):
    
    seeds = [0, 1, 2, 3, 4]
    repeat_metrics = []
    repeat_df = []
    repeat_records = []
    for seed in seeds:
        args.random_seed = seed
        metrics, df, records = cross_test( args, X, V, y, seed, flag)
        repeat_metrics.append(metrics)
        repeat_df.append(df)
        repeat_records.append(records)
    
    return repeat_metrics, repeat_df, repeat_records

def types_test( args, X, V, y):
    types = ["AddM",  "EPLM", "EPL"]
    X_base = X.copy()
    V_base = V.copy()
    single_flags = [[t] for t in types]
    pair_flags = [list(c) for c in combinations(types, 2)]
    triple_flags = [list(c) for c in combinations(types, 3)]
    all_flags = single_flags + pair_flags + triple_flags
    # all_flags = [["AddM",  "EPLM", "EPL"]]
    flag_result = {}
    mean_metric = {}
    for flag in all_flags:
        n_views = len(flag)
        flag_name = "_".join(flag)
        if n_views == 1:
            X_cur = [X_base.copy()]
            V_cur = [V_base.copy()]
            args.multi_view = False
        else:
            X_cur = [X_base.copy() for _ in range(n_views)]
            V_cur = [V_base.copy() for _ in range(n_views)]
            args.multi_view = True
        torch.manual_seed(2024)
        np.random.seed(2024)
        repeat_metrics, repeat_df, repeat_records = repeat_CV(args, X_cur, V_cur, y, flag)
        flag_result[flag_name] = {
            "flag": flag,
            "metrics": repeat_metrics,
            "df": repeat_df,
            "records": repeat_records,
            'mean_metric':np.vstack(repeat_metrics).mean(axis = 0),
            'std_metric':np.vstack(repeat_metrics).std(axis=0, ddof=1),
        }
        mean_metric[flag_name] = np.vstack(repeat_metrics).mean(axis=0)

    return flag_result, mean_metric
    
    

dataset = ['DBS', 'BCC', 'CPU', 'ESL', 'MPG', 'MMG', 'LEV', 'ERA', 'CEV']
# os.makedirs("../testdata/repeatCV/benchmark", exist_ok=True)
# for data in dataset:
#     args.dataset = data
#     X, V, y = MV_benchmark(args.dataset, args.gama)
#     flag_result , mean_metric= types_test( args, X, V, y)
#     data_to_save = {
#         'args': vars(args),
#         'flag_result': flag_result,
#         'mean_metric': mean_metric,
#         'X':X,
#         'V':V,
#         'y':y}
#     savepath = f"../testdata/repeatCV/benchmark/{args.dataset}.joblib"
#     joblib.dump(data_to_save, savepath, compress=3)

# flag_result = types_test( args, X, V, y)

X, V, y = MV_benchmark(args.dataset, args.gama)
flag_result , _= types_test( args, X, V, y)
benchmar_means = {}
benchmark_results = {}
for data in dataset:
    result = joblib.load(f"../testdata/repeatCV/benchmark/{data}.joblib")
    benchmar_means[data] = result['mean_metric']
    benchmark_results[data] = result['flag_result']
    
from scipy.stats import wilcoxon

def summarize_benchmark_results(benchmark_results, metric_idx=0):
    rows = []

    for dataset_name, flag_result in benchmark_results.items():
        single_flags = []
        pair_flags = []
        triple_flags = []

        # 按 flag 长度分类
        for flag_name, res in flag_result.items():
            n_views = len(res["flag"])
            if n_views == 1:
                single_flags.append(flag_name)
            elif n_views == 2:
                pair_flags.append(flag_name)
            elif n_views == 3:
                triple_flags.append(flag_name)

        # 找 best single
        best_single_name = max(
            single_flags,
            key=lambda name: flag_result[name]["mean_metric"][metric_idx]
        )
        best_single_res = flag_result[best_single_name]

        # 找 best pair
        best_pair_name = max( pair_flags, key=lambda name: flag_result[name]["mean_metric"][metric_idx])
        best_pair_res = flag_result[best_pair_name]

        # 找 triple（通常只有一个）
        best_triple_name = max(
            triple_flags,
            key=lambda name: flag_result[name]["mean_metric"][metric_idx]
        )
        best_triple_res = flag_result[best_triple_name]

        # 提取25个fold值
        single_vec = np.vstack(best_single_res['metrics'])[:, metric_idx]
        pair_vec = np.vstack(best_pair_res['metrics'])[:, metric_idx]
        triple_vec = np.vstack(best_triple_res['metrics'])[:, metric_idx]

        # Wilcoxon 检验
        try:
            p_pair_vs_single = wilcoxon(pair_vec, single_vec).pvalue
        except ValueError:
            p_pair_vs_single = np.nan

        try:
            p_triple_vs_pair = wilcoxon(triple_vec, pair_vec).pvalue
        except ValueError:
            p_triple_vs_pair = np.nan

        try:
            p_triple_vs_single = wilcoxon(triple_vec, single_vec).pvalue
        except ValueError:
            p_triple_vs_single = np.nan

        rows.append({
            "dataset": dataset_name,

            "best_single_name": best_single_name,
            "best_single_mean": best_single_res["mean_metric"][metric_idx],
            "best_single_std": best_single_res["std_metric"][metric_idx],

            "best_pair_name": best_pair_name,
            "best_pair_mean": best_pair_res["mean_metric"][metric_idx],
            "best_pair_std": best_pair_res["std_metric"][metric_idx],

            "triple_name": best_triple_name,
            "triple_mean": best_triple_res["mean_metric"][metric_idx],
            "triple_std": best_triple_res["std_metric"][metric_idx],

            "p_pair_vs_single": p_pair_vs_single,
            "p_triple_vs_pair": p_triple_vs_pair,
            "p_triple_vs_single": p_triple_vs_single,

            # "sig_pair_vs_single": p_to_dagger(p_pair_vs_single) if pd.notna(p_pair_vs_single) else "",
            # "sig_triple_vs_pair": p_to_ddagger(p_triple_vs_pair) if pd.notna(p_triple_vs_pair) else "",
        })

    return pd.DataFrame(rows)

def compare_best_vs_all(benchmark_results, metric_idx=2):
    """
    对每个数据集：
    1. 找到某个指标上 mean_metric 最优的组合
    2. 将该 best 组合与其他所有组合做 Wilcoxon signed-rank test
    3. 返回详细比较结果和每个数据集的 best 汇总
    """
    detail_rows = []
    summary_rows = []

    for dataset_name, flag_result in benchmark_results.items():
        all_flag_names = list(flag_result.keys())

        # 找 best 组合
        best_flag = max(
            all_flag_names,
            key=lambda name: flag_result[name]["mean_metric"][metric_idx]
        )
        best_res = flag_result[best_flag]

        best_vec = np.vstack(best_res["metrics"])[:, metric_idx]
        best_mean = best_res["mean_metric"][metric_idx]
        best_std = best_res["std_metric"][metric_idx]

        # 与其他组合逐一比较
        n_sig = 0
        for other_flag in all_flag_names:
            if other_flag == best_flag:
                continue

            other_res = flag_result[other_flag]
            other_vec = np.vstack(other_res["metrics"])[:, metric_idx]
            other_mean = other_res["mean_metric"][metric_idx]
            other_std = other_res["std_metric"][metric_idx]

            try:
                p_value = wilcoxon(best_vec, other_vec).pvalue
            except ValueError:
                p_value = np.nan

            mean_diff = best_mean - other_mean
            if pd.notna(p_value) and p_value < 0.1:
                n_sig += 1

            detail_rows.append({
                "dataset": dataset_name,
                "best_flag": best_flag,
                "other_flag": other_flag,
                "best_mean": best_mean,
                "best_std": best_std,
                "other_mean": other_mean,
                "other_std": other_std,
                "mean_diff": mean_diff,
                "p_value": p_value,
            })

        summary_rows.append({
            "dataset": dataset_name,
            "best_flag": best_flag,
            "best_mean": best_mean,
            "best_std": best_std,
            "n_compared": len(all_flag_names) - 1,
            "n_p_less_0.1": n_sig,
        })

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)

    return detail_df, summary_df
# summary_df_acc = summarize_benchmark_results(benchmark_results, metric_idx=0)
# summary_df_f1 = summarize_benchmark_results(benchmark_results, metric_idx=2)
detail_df_acc, summary_all_df_acc = compare_best_vs_all(benchmark_results, metric_idx=0)