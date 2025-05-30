import os
import sys
sys.path.append(os.getcwd())
import torch,time,datetime
from src import myGAE,myDAG_GNN,myGOLEM,myGraNDAG,myNotears_prior,mydagma_prior,varsortability
from rich import print as rprint
import numpy as np 
import pandas as pd

from casual_discovery.preparation import *
from prior_knowledge.knowledge_matrix_convert import *
from casual_discovery.evaluation import count_accuracy,numerical_SHD,sid



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type='gpu'
# device = torch.device("cpu")
# device_type='cpu'
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=4)

def main(output_path):
    dataset=''
    args={
        'alg':'dagma_prior',
        'dataset':'child',
        'proportion': 1,
        'size':2,
        'random':1,
        'random_seed':5,
        'LLM':'deepseek-r1'
        }    #方法
    args = get_config(args)
    

    del args.ER
    del args.method
    del args.n_nodes
    del args.sem_type
    del args.n_edges
    del args.adaptive_degree
    del args.scale
    rprint(vars(args))
    prior_index=3

    dataset=args.dataset
    size=args.size
    randomset=args.random
    true_dag=np.loadtxt(f'data_structure/{dataset}/{dataset}_graph.txt', delimiter=' ')
    X=pd.read_csv(f'dataset/data/{dataset}/{dataset}_continues_{size}dsize_random{randomset}.csv',dtype=float)
    X=X.dropna()  #删除含有nan的行
    varname=list(X.columns)

    X = X.to_numpy()
    if args.proportion>0:
        w_prior=np.loadtxt(f'prior_knowledge/prior_based_on_ground_truth/{dataset}/{dataset}.txt',dtype=int)
        #w_prior=np.loadtxt(f'prior_knowledge/prior_based_on_LLM/{dataset}/{dataset}_{model}.txt',dtype=int)
    else:
        w_prior=np.zeros([X.shape[1],X.shape[1]])

    sigma=1.0
    lambda1=0.01

    if args.alg=='gae':
        X=torch.tensor(X,dtype=torch.float64).to(device)
        model = myGAE(device_type=device_type,input_dim=10,update_freq=600)
    elif args.alg=='daggnn':
        X=torch.tensor(X,dtype=torch.float64).to(device)
        model = myDAG_GNN(device_type=device_type,k_max_iter=10,epochs=100)
    elif args.alg=='GOLEM':
        #不需要转换成tensor
        model = myGOLEM(device_type=device_type,num_iter=1e4)
    elif args.alg=='GraNDAG':
        #不需要转换成tensor
        model = myGraNDAG(device_type=device_type,input_dim=X.shape[1],iterations=1000,mu_init=1)
    elif args.alg in ['notears_prior', 'notears_quasi', 'notears_quasi_large']:
        model = myNotears_prior(lambda1=lambda1, sigma=sigma, loss_type='pdf')
    elif args.alg=='dagma_prior':
        lambda1=0.2
        model = mydagma_prior(loss_type='l2')
    elif args.alg=='varsortability':
        model = varsortability(type_='std')
    
    model.load_prior(w_prior,args.confidence)

    model.load_l1_penalty_parameter(lambda1)
    time1=time.time()
    model.learn(X)    

    
    metric = count_accuracy(true_dag,model.causal_matrix)
    
    time2=time.time()
    metric.update(sid(true_dag,model.causal_matrix))
    metric['time']=round(time2-time1,4)
    metric['lambda1']=round(lambda1,4)
    metric['sigma']=round(sigma,4)
    metric['finished']=datetime.datetime.now()
    if args.alg in ['notears_prior_order']:
        metric['score']=model.score
    rprint(metric)

    print(model.causal_matrix)
    for i in range(len(varname)):
        for j in range(len(varname)):
            if model.causal_matrix[i,j]==1:
                print(varname[i],varname[j])

    parameter={'size':args.size,'data':dataset, 
               'prior_proportion':args.proportion,'alg':args.alg,'random':args.random}
    parameter_values = [str(i) for i in parameter.values()]
    metric_values = [str(i) for i in metric.values()]

    if not os.path.exists(output_path):
        with open(output_path, 'a') as f:
            f.write(','.join(list(parameter.keys()))+',' +
                    ','.join(list(metric.keys()))+'\n')
    with open(output_path, 'a') as f:
        eva_info = ','.join(parameter_values)+','+','.join(metric_values)
        f.write(f'{eva_info}\n')

if __name__ == '__main__':

    main('out/output.csv')
    os._exit(-1)