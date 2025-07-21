# ASPBench: Can LLMs Solve ASP Problems? Insights from a Benchmarking Study

This is the code for the paper "Can LLMs Solve ASP Problems? Insights from a Benchmarking Study".


# Install Enviroment
```shell
conda create -n SymTex python=3.11
conda activate SymTex

pip install -r requirements.txt

conda install -c potassco clingo
```

Additionally, download the appropriate version of the DLV2 ASP solver from [https://dlv.demacs.unical.it/home](https://dlv.demacs.unical.it/home). After downloading, configure the path to the DLV2 executable in the `configs/dlv2.yaml` file.

# Dataset Generation Pipeline

This section outlines the step-by-step process to generate the SymTex dataset. Each step corresponds to a script execution.

0.  **Build ConceptNet Graph (Optional but Recommended):** Construct the ConceptNet graph, which is used for predicate modification in later steps.
    ```bash
    python 00_build_conceptnet_graph.py
    ```
1.  **Generate Raw Data Entries:** Create the initial set of raw data samples. Use the parallel script for faster generation on multi-core systems.
    ```bash
    # For parallel generation (recommended)
    bash scripts/run_symtex_raw_query_generation_parallel.sh 
    # Or, for sequential generation
    bash scripts/run_symtex_raw_query_generation.sh
    ```
2.  **Merge Raw Data:** Combine the raw data files generated in step 1 (especially if generated in parallel) into a single file.
    ```bash
    bash scripts/run_merge_raw_symtex.sh
    ```
3.  **Clean Raw Data:** Process the merged raw data from step 2 to remove structurally similar duplicate entries.
    ```bash
    bash scripts/clean_raw_data.sh
    ```
4.  **Filter Clean Data:** Apply filters to the cleaned data from step 3 based on specified criteria (e.g., minimum probability thresholds) to select high-quality samples.
    ```bash
    bash scripts/filter_clean_data.sh
    ```
5.  **Convert to DLV2 Format & Modify:** Transform the filtered data from step 4 into the DLV2 input format required by the ASP solver. This step may involve modifications to predicates and constants, potentially utilizing the ConceptNet graph built in step 0.
    ```bash
    bash scripts/run_dlv2_conversions.sh
    ```
6.  **Build ASP entailment Sub-dataset:** Construct the specific dataset tailored for the ASP entailment task using the data from step 5.
    ```bash
    bash scripts/run_fact_state_querying_construction.sh
    ```
7.  **Build Answer set verification Sub-dataset:** Construct the specific dataset tailored for the Answer set verification task using the data from step 5.
    ```bash
    bash scripts/run_answerset_selection_construction.sh
    ```
8.  **Build Answer set computation Sub-dataset:** Construct the specific dataset tailored for the Answer set computation task using the data from step 5.
    ```bash
    bash scripts/run_answerset_generation_construction.sh
    ```
9.  **Assemble Final SymTex Dataset:** Combine the sub-datasets generated in steps 6, 7, and 8 into the final SymTex benchmark dataset.
    ```bash
    bash scripts/run_construct_symtex.sh
    ```
10. **Locate Final Dataset:** The complete SymTex dataset assembled in step 9 will be available in the specified directory.
    *   Output Directory: [`datasets/symtex_final`](datasets/symtex_final)
11. **Textualize Final Dataset (Optional):** Convert the symbolic facts and rules in the final dataset (from step 9) into natural language descriptions. This uses the `TextulizationFramework` and saves results to a new directory.
    ```bash
    python 07_05_symtext_textulization.py --input-dir datasets/symtex_final --output-dir datasets/symtex_final_textual
    ```
    *   Textualized Output Directory: [`datasets/symtex_final_textual`](datasets/symtex_final_textual)
12. **Calculate Dataset Statistics:** Run the provided script to compute and display statistical information about the final **symbolic** SymTex dataset generated in step 9. The results will be saved in the [`experiments/symtex_statistic`](experiments/symtex_statistic) directory.
    ```bash
    python 08_statistic_symtex.py
    ```


# Additional Rule Types Explanation

In the dataset generation process (`src/dataset_generation/symtex.py`), five types of additional rules can be added, labeled 'a' through 'e'. These labels correspond to specific structural patterns in the generated Answer Set Programming (ASP) rules. Let `P_target` represent the target predicate of the original rule graph without these additional rules, `P_existing` represent any other predicate already existing in the original rule graph (excluding original facts when in the rule body), and `P_new` represent a predicate newly created during the additional rule generation process.

Here's a breakdown of each additional rule type:

| Label | Head Structure | Body Structure                     | Description                                                                 | Example (DLV2 format)                                    |
| :---- | :------------- | :--------------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------- |
| **a** | `P_target`     | Contains only `P_existing`         | The rule head is the target predicate, body uses existing predicates.       | `p_target(V1) :- p_existing1(V1), p_existing2(V1).`      |
| **b** | `P_existing`   | Must contain `P_target`, may have `P_existing` | The rule head is an existing predicate, body must include the target. | `p_existing1(V1) :- p_target(V1), p_existing2(V1).`      |
| **c** | `P_existing`   | Contains only `P_existing` (excluding head) | Both head and body use existing predicates (head != body predicates). | `p_existing1(V1) :- p_existing2(V1), p_existing3(V1).`      |
| **d** | Mix of `P_existing` / `P_new` | Mix of `P_existing` / `P_new` | Head and body are a mix of existing and newly created predicates.     | `p_new1(V1) :- p_existing1(V1), p_new2(V1).` <br> `p_existing1(V1) :- p_new1(V1), p_new2(V1).` |
| **e** | `P_new`        | Contains only `P_new` (excluding head) | Both head and body consist entirely of newly created predicates.        | `p_new1(V1) :- p_new2(V1), p_new3(V1).`                   |

**Note:**
*   Variables (like `V1`) are assigned based on the predicates' arity and connectivity.
*   Negations (strong `-` or default `not`) can also be applied to predicates based on generation probabilities, adding further complexity. The examples above omit negations for simplicity.
*   "Existing predicates" in the body generally exclude predicates that were originally facts in the original rule graph without these additional rules to avoid trivial rules.

# Note
A limitation exists: `min fact for query`, `additional facts`, and `additional rules` cannot all be active at the same time. You may only use a maximum of two from this group together.

# Evaluation
## Install Experimental Environment
```shell
conda activate SymTex

cd apiHelper
bash install.sh
```

## Run Evaluation Scripts
### ASP entailment Evaluation
Run the following command to evaluate the ASP entailment task:
```shell
bash scripts/run_fact_state_querying_evaluation.sh
```

### Answer set verification Evaluation
Run the following command to evaluate the Answer set verification task:
```shell
bash scripts/run_answer_set_decision_evaluation.sh
```

### Answer set computation Evaluation
Run the following command to evaluate the Answer set computation task:
```shell
bash scripts/run_answer_set_generation_evaluation.sh
```

The evaluation results will be saved in the following directories:
- ASP entailment: `experiments/fact_state_querying/w_few_shot/`
- Answer set verification: `experiments/answer_set_decision/w_few_shot/`
- Answer set computation: `experiments/answer_set_generation/w_few_shot/`

### Run All Evaluations
To run all evaluation tasks (ASP entailment, Answer set computation, and Answer set verification) sequentially, use the combined script:
```shell
bash scripts/predicate_and_evaluate/run_all_evaluations.sh
```
This script will execute the individual evaluation scripts in order and report the total execution time.

### Results
The `experiments/` directory stores the raw experimental data, including the inputs provided to LLMs and their corresponding outputs.
The `results/` directory contains detailed overall statistical data from the experiments.
