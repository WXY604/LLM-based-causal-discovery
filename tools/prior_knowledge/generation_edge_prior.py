import numpy as np
import random

def extract_prior_matrix(causal_matrix, sample_ratio=0.4):
    n = causal_matrix.shape[0]
    prior_matrix = np.zeros((n, n), dtype=int)
    
    # 收集所有可能的边
    edges = set()
    
    for i in range(n):
        for j in range(n):
            if causal_matrix[i, j] == -1:
                # 转换为(j,i)=1的形式
                edges.add((j, i))
            elif causal_matrix[i, j] == 1:
                edges.add((i, j))
    
    # 转换为列表以便抽样
    edge_list = list(edges)
    total_edges = len(edge_list)
    sample_size = int(total_edges * sample_ratio)
    
    # 随机抽样
    sampled_edges = random.sample(edge_list, sample_size)
    
    # 构建先验矩阵
    for (i, j) in sampled_edges:
        prior_matrix[i, j] = 1
    
    return prior_matrix

def save_matrix_to_txt(matrix, filename):
    with open(filename, 'w') as f:
        for row in matrix:
            row_str = ' '.join(map(str, row))
            f.write(row_str + '\n')


if __name__ == "__main__":
    datasets = ["alarm","child"]  # 添加你的所有数据集名称
    models = ["deepseek-r1", "gpt-4o"]  # 添加你的所有模型名称
    # 遍历所有组合
    for dataset in datasets:
        for model in models:
            knowledge_matrix = np.loadtxt(f'prior_knowledge/LLM_knowledge/{dataset}/{dataset}_{model}.txt', delimiter=' ')
            prior_matrix = extract_prior_matrix(knowledge_matrix)
            save_matrix_to_txt(prior_matrix, f'prior_knowledge/prior_based_on_LLM/{dataset}/{dataset}_{model}.txt')
        knowledge_matrix = np.loadtxt(f'data_structure/{dataset}/{dataset}_graph.txt', delimiter=' ')
        prior_matrix = extract_prior_matrix(knowledge_matrix)
        save_matrix_to_txt(prior_matrix, f'prior_knowledge/prior_based_on_ground_truth/{dataset}/{dataset}.txt')
