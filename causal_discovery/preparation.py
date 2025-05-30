'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-01-04 15:37:38
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-05-02 14:45:47
FilePath: /IJCNN/preparation.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
import os
import numpy as np
import random

def get_config(set_arg=None):
    parser = argparse.ArgumentParser(description='experiments on various ')
    # help Display description information for candidate parameters
    parser.add_argument('--n_nodes', type=int, default=20, help='')
    parser.add_argument('--ER', type=int, default=2, help='')
    parser.add_argument('--size', type=int, default=2, help='')
    parser.add_argument('--graph_type', type=str, default='ER', help='')
    parser.add_argument('--random', type=int, default=0, help='')
    parser.add_argument('--random_seed', type=int, default=0, help='')


    parser.add_argument('--method', type=str, default='linear', help='')
    parser.add_argument('--sem_type', type=str, default='gauss', help='')
    
    parser.add_argument('--prior_type', type=str, default='exist', help='')
    parser.add_argument('--confidence', type=float, default=0.9, help='')
    parser.add_argument('--proportion', type=float, default=0.9, help='')
    parser.add_argument('--error_prior_proportion', type=float, default=0., help='')
    parser.add_argument('--error_prior_type', type=str, default='reverse direct', help='')

    parser.add_argument('--alg', type=str, default='notears', help='')
    parser.add_argument('--scale', type=str, default='no', help='')
    parser.add_argument('--adaptive_degree', type=float, default=1, help='')

    parser.add_argument('--test', type=int, default=1, help='')

    parser.add_argument('--dataset', type=str, default='', help='')
    parser.add_argument('--LLM', type=str, default='', help='')

    args = parser.parse_args()

    if set_arg != None and args.test: 
        for key in set_arg:
            setattr(args, key, set_arg[key])
    
    args.n_edges=args.ER*args.n_nodes
    return args

def get_config_real(set_arg=None):
    parser = argparse.ArgumentParser(description='experiments on various ')
    # help Display description information for candidate parameters
    
    parser.add_argument('--dataset', type=str, default='sachs', help='')

    parser.add_argument('--prior_type', type=str, default='exist', help='')
    parser.add_argument('--confidence', type=float, default=0.9, help='')
    parser.add_argument('--proportion', type=float, default=0.9, help='')
    parser.add_argument('--error_prior_proportion', type=float, default=0., help='')
    parser.add_argument('--error_prior_type', type=str, default='reverse direct', help='')

    parser.add_argument('--alg', type=str, default='notears', help='')

    parser.add_argument('--test', type=int, default=1, help='')
    args = parser.parse_args()

    if set_arg != None and args.test: 
        for key in set_arg:
            setattr(args, key, set_arg[key])
    
    return args

def normalize(X,normalize_type):
    std_devs = np.std(X, axis=0, ddof=0)  # ddof=0表示标准差为总体标准差
    if normalize_type=='no':
        X_normalized = X
    elif normalize_type=='std':
        X_normalized = X / std_devs
    elif normalize_type=='part':
        X_normalized = X / (std_devs**0.5)
    return X_normalized


