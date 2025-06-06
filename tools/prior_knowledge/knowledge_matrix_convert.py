'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-04-13 16:45:25
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-05-30 03:23:32
FilePath: /IJCNN/generation/generation_prior.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import glob

def read_mapping_file(mapping_file):
    with open(mapping_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def generate_relation_matrix(folder_path, mapping):
    # 创建文件名到索引的映射（大小写不敏感）
    name_to_index = {name.lower(): idx for idx, name in enumerate(mapping)}
    n = len(mapping)
    
    # 初始化N×N矩阵
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    # 处理每个文件
    for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
        with open(file_path, 'r') as f:
            content = f.read()
        
        current_file = os.path.basename(file_path).split('.')[0]
        if current_file.lower() not in name_to_index:
            continue  # 跳过不在mapping中的文件
        
        current_idx = name_to_index[current_file.lower()]
        
        # 解析文件内容
        sections = {}
        current_section = None
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.lower().startswith('before:'):
                current_section = 'before'
                sections[current_section] = []
            elif line.lower().startswith('after:'):
                current_section = 'after'
                sections[current_section] = []
            elif line.lower().startswith('not confirm:'):
                current_section = 'not_confirm'
                sections[current_section] = []
            elif current_section:
                name = line.split('.')[0]
                sections[current_section].append(name.lower())
        
        # 填充矩阵
        for section in ['before', 'after']:
            if section in sections:
                for name in sections[section]:
                    if name in name_to_index:
                        target_idx = name_to_index[name]
                        matrix[current_idx][target_idx] = -1 if section == 'before' else 1
    
    return matrix

def write_matrix(matrix, output_file):
    with open(output_file, 'w') as f:
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')

import os

def process_combination(dataset, model):
    folder_path = f"LLM_query/LLM_answer/{dataset}/{model}"
    mapping_file = f"data_structure/{dataset}/{dataset}.mapping"
    
    # 读取mapping文件
    try:
        mapping = read_mapping_file(mapping_file)
        print(f"读取到变量顺序: {mapping}")
    except FileNotFoundError:
        print(f"错误: 找不到mapping文件 {mapping_file}")
        return
    
    # 生成矩阵
    matrix = generate_relation_matrix(folder_path, mapping)
    
    # 写入矩阵到文件
    output_file = f"prior_knowledge/LLM_knowledge/{dataset}/{dataset}_{model}.txt"
    write_matrix(matrix, output_file)
    print(f"矩阵已保存到 {output_file}")

if __name__ == "__main__":
    # 定义所有要处理的数据集和模型组合
    datasets = ["alarm","child"]  # 添加你的所有数据集名称
    models = ["deepseek-r1", "gpt-4o"]  # 添加你的所有模型名称
    
    # 遍历所有组合
    for dataset in datasets:
        for model in models:
            print(f"\n正在处理组合: 数据集={dataset}, 模型={model}")
            try:
                process_combination(dataset, model)
            except Exception as e:
                print(f"处理组合 {dataset}-{model} 时出错: {str(e)}")
                continue