
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


## **Framework Overview**


![Figure1.](images/framework.PNG)

The above framework diagram provides a more intuitive understanding of the entire workflow.

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



## **Conclusion and Outlook**

We expect this toolkit to provide effective support for professionals engaged in causal science research and practice, helping users leverage the powerful knowledge source of Large Language Models more conveniently and reliably. We believe this approach offers a valuable technical direction for performing causal discovery in a stable and cost-effective manner and look forward to promoting the further development of this field in collaboration with both academia and industry.



Of course, here is the translation of the Chinese parts into English, with the original structure and file names preserved.

# Dictionary



## Dataset Part

#### `data_structure`

- `{Dataset_name}`

    - `{Dataset_name}_graph.txt`：The ground truth causal graph of the dataset variables.

    - `{Dataset_name}.mapping`：Mapping of dataset variable names.

#### `dataset`

- `{data}`

    - `{Dataset_name}`

        - `{Dataset_name}_continues_{n}dsize_random{r}`：Synthetic datasets, n represents the ratio of dataset size to the number of variables, r represents random generation parameters.



## LLM Part



#### `prompt_design`

-   `description`

    -   `{Dataset_name}.json`：Explanation of variables in the dataset.

-   `prompt_generation.py`：Generate the required prompt based on the content in the description.

-   `prompt`

    -   `{Dataset_name}`

        -   `{Dataset_name}_{Variable_name}.txt`：The actual prompt used.

#### `LLM_query`

-   `api.py`：Call the API to inquire about the causal relationships of dataset variables.

-   `LLM_answer`

    -   `{Dataset_name}`

        -   `{LLM_name}`

            -   `{Variable_name}.txt`：The causal relationships (causes and effects) of a specific variable in the dataset.



## Causal Discovery Part



#### `prior_knowledge`

-   `knowledge_matrix_convert.py`：Clean the knowledge provided by the large model and convert it into matrix form.

-   `LLM_knowledge`

    -   `{Dataset_name}`

        -   `{Dataset_name}_{LLM_name}.txt`：Large model knowledge stored in matrix form.

-   `generation_edge_prior.py`：Generate edge priors based on large model knowledge or the ground truth causal graph.

-   `prior_based_on_LLM`

    -   `{Dataset_name}`

        -   `{Dataset_name}_{LLM_name}.txt`：Edge prior matrix generated based on the large language model.

-   `prior_based_on_ground_truth`

    -   `{Dataset_name}`

        -   `{Dataset_name}.txt`：Edge prior matrix generated based on the ground truth causal graph.

#### `src`

-   `{method_name}.py`：Main method for causal discovery.

#### `causal_discovery`

-   `evalution.py`：Evaluation functions used during the training process.

-   `preparation.py`：Preparatory work such as parameter setting.

-   `main.py`：Main program.

#### `out`

-   `output.csv`：Display of various parameters and metrics of the model training results.
 
# Usage

## **Installation Guide for LLM-based-causal-discovery**

This guide provides step-by-step instructions for downloading the `LLM-based-causal-discovery` project and setting up its local environment.

Follow these steps to get the project running.

#### **Step 1: Clone the GitHub Repository**

Open your terminal (Command Prompt or PowerShell on Windows, Terminal on macOS/Linux) and use the `git clone` command to download the project source code.

```bash
git clone https://github.com/WXY604/LLM-based-causal-discovery.git
```

This will create a folder named `LLM-based-causal-discovery` in your current directory. Navigate into this new folder:

```bash
cd LLM-based-causal-discovery
```

#### **Step 2: Create and Activate a Python Virtual Environment**

Using a virtual environment is highly recommended to keep project dependencies isolated.

* **Create the virtual environment** (we'll name it `venv`):

    ```bash
    python -m venv venv
    ```

* **Activate the virtual environment**:
    * **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS and Linux**:
        ```bash
        source venv/bin/activate
        ```

    Once activated, you will see `(venv)` at the beginning of your terminal prompt.

#### **Step 3: Install Project Dependencies**

All required Python libraries are listed in the `Requirement.txt` file. You can install them all with a single `pip` command.

Make sure your virtual environment is activated, then run:

```bash
pip install -r Requirement.txt
```

`pip` will automatically read the file and install all necessary packages. This process may take a few moments.

## Run


