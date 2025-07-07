
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


# Several LLM-based Methods Can Be Used

This section introduces some methods for discovering and generating priors using large language models, for reference.

## LLM-Driven Causal Discovery via Harmonized Prior

The core idea of this framework is to use an LLM as a knowledge expert. Through specially designed Prompting Strategies, it guides the LLM to perform causal reasoning from two different yet complementary perspectives to generate a reliable "Harmonized Prior". This harmonized prior is then integrated into mainstream causal structure learning algorithms to enhance the accuracy and reliability of discovering causal relationships from data.

<img src="images/Harmonized_1.PNG" alt="Harmonized_1" width="300" align="left" hspace="15" vspace="5">

The overall logical flow of the framework is shown in the left figure and primarily consists of three parts: Dual-Expert LLM Reasoning, Harmonized Prior Construction, and Plug-and-Play Structure Learning.



**1. Dual-Expert LLM Reasoning Module**
To ensure the accuracy of the causal knowledge provided by the LLM, this framework does not have the LLM directly judge the complex relationships between all pairs of variables. Instead, it configures the LLM into two different expert roles focused on specific tasks: the Conservative Expert and the Exploratory Expert.

* **Conservative Expert - Aims for Precision**
    * As shown in the left figure, the goal of the Conservative Expert is to identify the most explicit and reliable causal relationships.
    * It first uses "single-step reasoning" to quickly screen for causal pairs with the highest confidence.
    * Subsequently, it employs a "Decomposition and Verification" strategy to meticulously verify and reconfirm these selected relationships one by one, in order to filter out potential spurious associations.
    * The final output is a high-precision set of causal relationships, $\lambda_p$, which is used as a "Path Existence" constraint. That is, if $(A,B)$ is in this set, it is believed that a path from A to B exists in the true causal graph.

<img src="images/Harmonized_2.PNG" alt="Harmonized_2" width="300" align="left" hspace="15" vspace="5">

* **Exploratory Expert - Aims for Recall**
    * As shown in the left figure, the goal of the Exploratory Expert is to identify all potential causal links as comprehensively as possible.
    * This module centers on each variable, analyzing one by one which other variables in the dataset could be its direct causes.
    * Through this "Decomposition and Exploration" approach, it generates a list of "possible causes" $C(x_i)$ for each variable.
    * All these possible causes are aggregated into a high-recall set of causal relationships, $\lambda_r$. This set is used to define an "Edge Absence" constraint, meaning if a causal relationship $(A,B)$ does not appear in this set, generating a direct edge from A to B in the final causal graph is forbidden.

<img src="images/Harmonized_3.PNG" alt="Harmonized_3" width="300" align="left" hspace="15" vspace="5">

**2. Harmonized Prior Construction**
The framework fuses the causal knowledge output by the two aforementioned experts to construct a unified "Harmonized Prior". This harmonized prior combines the advantages of both:

* **Path Existence Constraint:** Utilizes the high-precision causal relationships $\lambda_p$ output by the Conservative Expert.
* **Edge Absence Constraint:** Utilizes the high-recall causal relationships $\lambda_r$ output by the Exploratory Expert to define the scope of possible direct edges.

In this way, it not only ensures that strong causal signals are not lost but also effectively constrains the search space for structure learning by ruling out a large number of impossible causal connections, thereby improving overall accuracy.

**3. Integration with Structure Learning Algorithms**
Finally, this constructed "Harmonized Prior" is integrated in a "plug-and-play" manner into various mainstream causal structure learning algorithms, as shown in the top box of Figure 1. Whether they are Score-based, Constraint-based, or Gradient-based methods, all can leverage this harmonized prior to guide their search process, ultimately learning a more accurate and reliable causal graph from observational data.



## From Query Tools to Causal Architects

This section will introduce another framework for leveraging Large Language Models (LLMs) for causal discovery. Unlike the aforementioned methods, this framework adopts a three-stage sequential prompting process, designed to extract and revise causal knowledge from an LLM, and then use this knowledge as prior information to guide traditional data-driven causal structure learning algorithms.

<img src="images/Query.PNG" alt="Query_1" width="500" align="right" hspace="15" vspace="5">

The core logic of the entire framework can be understood through the example prompts in the table on the right, which clearly demonstrates the three core stages: Variable Understanding, Causal Discovery, and Error Revision.


**1. Stage One: Variable Understanding**

The objective of this stage is to first have the LLM accurately understand the real-world meaning of each variable in the dataset.

* **Input:** The researcher provides the LLM with the symbol and possible values for each variable. This is typically the most basic information available in a standard dataset.
* **Example Prompt ("Prompt Understand"):** As shown in Table 1, the prompt asks the LLM to act as an expert in a specific domain and explain the meaning of each variable based on its symbol and values.
* **Output:** The LLM generates a detailed textual description for each variable, laying the foundation for a subsequent causal inference.

**2. Stage Two: Causal Discovery**

Based on a full understanding of the variables' meanings, the LLM is asked to identify the causal relationships among them.

* **Example Prompt ("Prompt Causal Discovery"):** The prompt for this stage requires the LLM to analyze the causal effects between variables and output the results in the form of a directed graph network.
* **Key Requirement:** The prompt here explicitly emphasizes that every edge in the graph must represent a **direct causal relationship** between the two variables.

**3. Stage Three: Error Revision**

To enhance the reliability of the LLM's output, the framework introduces a self-revision stage, prompting the LLM to check and correct its own previously generated conclusions.

* **Example Prompt ("Prompt Revision"):** The causal statements generated in the second stage (e.g., $x_i \rightarrow x_j$) are used as input, asking the LLM in return whether these statements are correct and requesting it to provide reasons.
* **Objective:** Through this self-checking mechanism, inaccurate causal statements are filtered out, ultimately yielding a higher-quality, more reliable set of causal relationships to guide the subsequent data analysis process.

**4. Integration with Data-Driven Algorithms**

A core insight of this framework is that although the LLM is prompted to output "direct" causal relationships, the knowledge inferred by the LLM is inherently closer to qualitative, indirect causal relationships. Therefore, the causal statements obtained through the three-stage process are not directly treated as the final causal graph.

* **Ancestral Constraints:** The framework converts the LLM's revised causal statements (e.g., A causes B) into ancestral constraints. This means that in the final causal graph, there must exist a directed path from A to B, but not necessarily a direct edge.
* **Hard and Soft Approaches:** These ancestral constraints are then integrated into score-based causal structure learning algorithms. Researchers can choose different integration strategies based on their level of confidence in the LLM's prior knowledge:
    * **Hard Constraint Approach:** Strictly enforces that the final causal graph must satisfy all ancestral constraints provided by the LLM, narrowing the search space through methods like pruning.
    * **Soft Constraint Approach:** Incorporates the LLM's prior knowledge as part of the scoring function. This method allows for discarding some prior constraints if there is a strong conflict between the data and the prior knowledge, thus possessing a degree of fault tolerance.


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

-----

# 🚀 Getting Started

This project can be used in two primary ways:

1.  **For End-Users:** By downloading and running the pre-built, standalone application (recommended).
2.  **For Developers:** By cloning the source code and setting up the local development environment.

-----

## For End-Users (Recommended Method)

The easiest way to use this system is to download the latest pre-built version for Linux. This does not require you to install Python or any dependencies manually, except for two system tools.

**1. Install Prerequisites**

This application requires `graphviz` (for graph rendering) and `unrar` (for extracting the downloaded file). Please install them first:

```bash
# For Debian/Ubuntu-based systems
sudo apt-get update && sudo apt-get install graphviz unrar
```

**2. Download the Application**

Go to the project's **[GitHub Releases Page](https://www.google.com/search?q=https://github.com/WXY604/LLM-based-causal-discovery/releases)** to find the latest version.

[](https://www.google.com/search?q=https://github.com/WXY604/LLM-based-causal-discovery/releases)

Download the `.rar` archive (e.g., `Causal.Discovery.System.part1.rar`) from the "Assets" section.

**3. Extract and Run**

Open a terminal and run the following commands:

```bash
# 1. Extract the RAR archive. Quotes are important due to spaces in the filename.
unrar x "Causal Discovery System.part1.rar"

# 2. Navigate into the new directory
cd CD

# 3. Grant execute permissions to the main program
chmod +x CD

# 4. Run the application
./CD
```

The program will start in the background and automatically open the user interface in your default web browser. To stop the application, press `Ctrl + C` in the terminal.

-----

## For Developers (From Source)

If you wish to run the scripts directly, modify the code, or contribute to the project, follow these instructions to set up a local development environment.

#### Installation Guide

This guide provides step-by-step instructions for downloading the `LLM-based-causal-discovery` project and setting up its local environment.

**Step 1: Clone the GitHub Repository**

Open your terminal and use the `git clone` command to download the project source code.

```bash
git clone https://github.com/WXY604/LLM-based-causal-discovery.git
```

This will create a folder named `LLM-based-causal-discovery`. Navigate into this new folder:

```bash
cd LLM-based-causal-discovery
```

**Step 2: Create and Activate a Python Virtual Environment**

Using a virtual environment is highly recommended to keep project dependencies isolated.

```bash
# Create the virtual environment (we'll name it venv):
python -m venv venv

# Activate the virtual environment:
# On Windows:
.\venv\Scripts\activate
# On macOS and Linux:
source venv/bin/activate
```

Once activated, you will see `(venv)` at the beginning of your terminal prompt.

**Step 3: Install Project Dependencies**

All required Python libraries are listed in the `requirements.txt` file. Make sure your virtual environment is activated, then run:

```bash
pip install -r requirements.txt
```

This process may take a few moments.

#### Runbook

This guide provides detailed instructions for running the causal discovery process in different scenarios from the source code.

**Scenario 1: Running Causal Discovery Algorithm Only**

If you already have all the pre-processed data and prior knowledge, you can run the main program directly.

```bash
python tools/causal_discovery/main.py
```

**Scenario 2: Causal Discovery from a Raw Dataset**

This workflow guides you through the entire process, from downloading a raw dataset to completing the discovery.

1.  **Prepare Ground Truth and Mapping Files:** Place these files into `data_structure/{Dataset_name}/`.
2.  **Prepare the Dataset Data File:** Place your `.csv` or `.txt` file into `dataset/data/{Dataset_name}/`.
3.  **Generate the Prior Matrix:** Run the script to generate the edge prior matrix from the ground truth.
    ```bash
    python tools/causal_discovery/prior_knowledge/generation_edge_prior.py
    ```
4.  **Run the Main Program:**
    ```bash
    python tools/causal_discovery/main.py
    ```

**Scenario 3: LLM-Assisted Causal Discovery**

This workflow uses external knowledge from a Large Language Model (LLM) as a prior.

1.  **Prepare Data:** Follow steps 1 & 2 from Scenario 2.
2.  **Obtain the LLM Knowledge Matrix:** Use the tools in `LLM_query` (or your own methods) and format the results as specified in `LLM_knowledge/{Dataset_name}/{Dataset_name}_{LLM_name}.txt`.
3.  **Generate Prior Matrix from LLM Knowledge:** First, **modify** the `tools/causal_discovery/prior_knowledge/generation_edge_prior.py` script to switch its logic to generate the prior based on LLM knowledge. After modifying, run it:
    ```bash
    python tools/causal_discovery/prior_knowledge/generation_edge_prior.py
    ```
4.  **Run the Main Program:**
    ```bash
    python tools/causal_discovery/main.py
    ```

