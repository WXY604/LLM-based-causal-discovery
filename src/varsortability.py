'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-01-04 15:37:38
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-04-16 16:56:32
FilePath: /IJCNN/src/varsortability.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from CausalDisco.analytics import (
    var_sortability,
    r2_sortability,
    snr_sortability
)

from CausalDisco.baselines import (
    random_sort_regress,
    var_sort_regress,
    r2_sort_regress,
    sort_regress
)

class varsortability():
    def __init__(self,type_='no'):
        self.type=type_
    def learn(self,X):
        if self.type=='no':
            self.weight_causal_matrix=var_sort_regress(X)
            self.weight_causal_matrix[np.abs(self.weight_causal_matrix) < 0.3] = 0
        elif self.type=='std':
            X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)
            self.weight_causal_matrix=var_sort_regress(X_std)
            self.weight_causal_matrix[np.abs(self.weight_causal_matrix) < 0.3] = 0
        elif self.type=='only_order':
            X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)
            self.weight_causal_matrix=sort_regress(X_std,np.var(X, axis=0))
            self.weight_causal_matrix[np.abs(self.weight_causal_matrix) < 0.3] = 0

        for i in range(self.weight_causal_matrix.shape[0]):
            for j in range(self.weight_causal_matrix.shape[1]):
                if self.w_prior[i, j] == 1 and self.weight_causal_matrix[i, j] == 0.0:
                    self.weight_causal_matrix[i, j] = 1
        self.causal_matrix=1*(self.weight_causal_matrix!=0)
        return self.causal_matrix
    
    def load_prior(self,w_prior=None, prob_prior=0):
        self.w_prior=w_prior
        self.prob_prior=prob_prior

    def load_l1_penalty_parameter(self,lambda1=0):
        self.penalty_lambda=lambda1