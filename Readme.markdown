# Dictionary


## Dataset Part
#### `data_structure`
- `{Dataset_name}`
    - `{Dataset_name}_graph.txt`：数据集变量的真实因果图
    - `{Dataset_name}.mapping`：数据集变量名对照
####  `dataset`
- `{data}`
    - `{Dataset_name}`
        - `{Dataset_name}_continues_{n}dsize_random{r}`：合成数据集，n表示数据集规模与变量数目的比值，r表示随机生成参数

## LLM Part

#### `prompt_design`
-   `description`
    -   `{Dataset_name}.json`：数据集中变量的解释
-   `prompt_generation.py`：根据description中的内容生成所需要的prompt
-   `prompt`
    -   `{Dataset_name}`
        -   `{Dataset_name}_{Variable_name}.txt`：实际使用的prompt
#### `LLM_query`
-   `api.py`：调用api对数据集变量进行因果关系询问
-   `LLM_answer`
    -   `{Dataset_name}`
        -   `{LLM_name}`
            -   `{Variable_name}.txt`：数据集中某个特定变量的前后因果关系

## Causal Discovery Part

####   `prior_knowledge`
-   `knowledge_matrix_convert.py`：将大模型提供的知识清洗并转化为矩阵形式
-   `LLM_knowledge`
    -   `{Dataset_name}`
        -   `{Dataset_name}_{LLM_name}.txt`：矩阵形式存储的大模型知识
-   `generation_edge_prior.py`：基于大模型知识或者真实因果图生成边先验
-   `prior_based_on_LLM`
    -   `{Dataset_name}`
        -   `{Dataset_name}_{LLM_name}.txt`：基于大语言模型生成的边先验矩阵
-   `prior_based_on_ground_truth`
    -   `{Dataset_name}`
        -   `{Dataset_name}.txt`：基于真实因果图生成的边先验矩阵
####   `src`
-   `{method_name}.py`：因果发现主方法
####   `causal_discovery`
-   `evalution.py`：训练过程中所使用的计算函数
-   `preparation.py`：参数设置等前置准备
-   `main.py`：主程序
####   `out`
-   `output.csv`：模型训练结果的各项参数与指标展示
 

