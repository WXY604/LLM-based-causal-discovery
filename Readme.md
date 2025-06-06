
# LLM-Augmented Causal Discovery Toolkit: A Technical Introduction

## **Background and Challenges**

Inferring causal relationships from observational data is a core challenge in data science and related research fields. Traditional causal discovery methods rely heavily on prior knowledge from domain experts for guidance. However, acquiring such knowledge often involves significant costs in terms of both time and money, which largely limits the application scope of advanced causal discovery techniques.

## **Opportunities and Challenges of Large Language Models**

The emergence of Large Language Models (LLMs) has provided new possibilities for acquiring prior knowledge. By querying LLMs about the relationships between variables, researchers can obtain judgments that approach the level of an expert. The advantage of this approach lies in significantly reducing the cost of knowledge acquisition. Furthermore, in some scenarios, the knowledge provided by LLMs can be more objective than the judgments of non-professionals.

However, this new paradigm is also accompanied by challenges. LLMs have inherent instability; queries on the same topic may return inconsistent or even self-contradictory results. Using this inaccurate or internally contradictory information directly as priors can not only fail to improve model performance but may also negatively impact the accuracy of the final analysis.

## **Our Approach and the Toolkit's Core Functionality**

To address this challenge, we have developed this toolkit. It aims to fully leverage the powerful knowledge base of LLMs while systematically mitigating the risks associated with their instability.

Our core approach is inspired by the latest research findings: guiding an LLM to determine the concrete temporal order of events yields more reliable and stable outputs compared to directly asking it to judge abstract causal relationships.

Based on this, the core functionalities of this toolkit include:

  * **Structured Knowledge Elicitation**: After the user defines the research scenario and variables, the toolkit automatically generates structured queries to guide the LLM, efficiently extracting high-confidence information regarding the temporal order of variables.
  * **Integration and Refinement of Model Outputs**: The toolkit includes a built-in analytical mechanism to process the initial information returned by the LLM. It systematically integrates these potentially inconsistent, localized judgments with the goal of refining a more globally consistent and reliable variable ordering.
  * **Compatibility with Downstream Algorithms**: The refined temporal priors produced by the toolkit can serve as high-quality constraints and be flexibly applied to various mainstream causal discovery algorithms, helping to construct more accurate and robust causal structures from real-world data.

-----

## **Framework Overview**


![Figure1.](images/framework.PNG)

The following framework diagram (Figure 1) provides a more intuitive understanding of the entire workflow.

The entire process can be summarized into several high-level stages:

  * **Stage 1: Initial Knowledge Generation (Partial Order Generation)**

      * In this stage, we initiate structured queries to the LLM through scenario simulation and metadata input.
      * The objective is to obtain the model's preliminary, discrete judgments about the temporal sequence of variables.

  * **Stage 2: Knowledge Integration and Refinement (Conflicting Decomposition & Optimal Total Order Discovery)**

      * This is a critical step in the process. The toolkit systematically analyzes all preliminary judgments obtained from the LLM.
      * It integrates these scattered and potentially inconsistent local pieces of information with the aim of refining a more globally consistent and reliable variable ordering.

  * **Stage 3: Guiding Downstream Analysis (Order-based Causality)**

      * Finally, this refined global ordering serves as a high-quality prior knowledge.
      * It can be input into any standard causal discovery algorithm chosen by the user, acting as a strong external guide to help the algorithm converge more accurately on the real data to infer the final causal graph.

In short, the core of this framework is to transform the potentially vague, contradictory, and localized knowledge provided by an LLM into a clear and reliable global variable ordering through a series of systematic steps, thereby providing effective support for data-driven causal learning.



-----

## **Conclusion and Outlook**

We expect this toolkit to provide effective support for professionals engaged in causal science research and practice, helping users leverage the powerful knowledge source of Large Language Models more conveniently and reliably. We believe this approach offers a valuable technical direction for performing causal discovery in a stable and cost-effective manner and look forward to promoting the further development of this field in collaboration with both academia and industry.



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
 
# Run
        -`python causal_discovery/main.py`
