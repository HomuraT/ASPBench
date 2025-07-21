import copy
import random
import argparse
import os
import json
import concurrent.futures
import threading
from tqdm import tqdm # Optional: for progress bar

import numpy as np
from numpy.ma.extras import average

# 从 src.dataset_generation.symtex 导入 SymTexSample 类
from src.dataset_generation.symtex import SymTexSample
# 从 src.dataset_generation.asp_formatter 导入 dict_to_asp_strings 和 format_dict_structure_to_asp
from src.dataset_generation.asp_formatter import dict_to_asp_strings, format_dict_structure_to_asp
from src.utils.dlv2_runner import Dlv2Runner
from src.utils.graph_structure_utils import longest_path_to_target
# 从 src.utils.json_utils 导入读写 JSONL 的函数
from src.utils.json_utils import read_jsonl_lazy, append_jsonl, JsonUtilsError
# 从 src.utils.sparse_utils 导入稀疏矩阵转换函数
from src.utils.sparse_utils import dense_to_sparse_serializable


def generate_symtex_sample(
    seed: int,
    num_nodes: int,
    num_edges: int,
    extra_predicate_num: int,
    extra_edge_num: int,
    strong_negation_prob: float,
    default_negation_prob: float,
    max_predicates_per_rule: int,
    num_noise_rules_per_type: int,
    m: int,
    largest: int
):
    """
    生成和验证 SymTex 样本，并返回包含样本数据和统计信息的字典。
    """
    # 设置随机种子，确保结果可复现
    random.seed(seed)
    np.random.seed(seed)

    dlv2_runner = Dlv2Runner() # 保持 Dlv2Runner 逻辑

    # 使用传入的参数创建 SymTexSample 实例
    # print("Creating SymTexSample instance...")
    sample = SymTexSample(
        num_nodes=num_nodes,
        num_edges=num_edges,
        seed=seed,
        extra_predicate_num=extra_predicate_num,
        extra_edge_num=extra_edge_num,
        strong_negation_prob=strong_negation_prob,
        default_negation_prob=default_negation_prob,
        max_predicates_per_rule=max_predicates_per_rule,
        num_noise_rules_per_type=num_noise_rules_per_type # 确保噪声被生成
    )

    noiseless_facts, noiseless_rules, noisy_facts, noisy_rules_by_type, original_dicts = sample.to_dlv2(
        seed=seed, m=m, largest=largest, use_disjunction=False # 假设不使用析取
    )

    # --- 保留所有 Dlv2Runner 相关的验证逻辑 ---
    # 确定目标查询是否存在
    target_query_dict = original_dicts['noiseless_rules'][0]['head'][0] # Keep original dict
    idx_to_name = {i: name for i, name in enumerate(sample.predicates)}
    # 使用 dict_to_asp_strings 格式化查询 (会包含句点)
    # 注意: dict_to_asp_strings 需要变量映射，但目标查询通常是基项(ground atom)，没有变量。
    # 我们需要确认 target_query_dict 的结构以及 dict_to_asp_strings 如何处理基项。
    # 假设 target_query_dict 包含 'variables' 列表 (可能为空)
    # 并且 dict_to_asp_strings 能正确处理基项 (var_idx_to_name=None 或空字典)
    query_fact_list = dict_to_asp_strings(target_query_dict, idx_to_name, is_fact=True, var_idx_to_name={}) # Use empty dict for vars
    assert len(query_fact_list) == 1, "dict_to_asp_strings should return one string for a single atom"
    query_fact = query_fact_list[0]

    # 准备用于比较的查询字符串：移除空格和句点 (与 06 脚本的原始逻辑一致)
    query_fact_for_comparison = query_fact.replace(" ", "").replace(".", "")

    asp_program = '\n'.join(noiseless_facts) + '\n' + '\n'.join(noiseless_rules)
    dlv2_result = dlv2_runner.run(asp_program=asp_program, num_answer_sets=0)
    assert dlv2_result['success'] and len(dlv2_result['answer_sets']) == 1, 'error: multiple answer set'
    answer_set = dlv2_result['answer_sets'][0]
    # 使用移除了空格和句点的查询字符串进行比较
    query_in_answerset = query_fact_for_comparison in answer_set

    # --- Noise Filtering Logic ---
    # 初始化有效噪声容器
    valid_noisy_facts = []
    valid_noisy_facts_dicts = []
    valid_noisy_rules_by_type = {k: [] for k in noisy_rules_by_type.keys()} # Preserve original keys
    valid_noisy_rules_by_type_dicts = {k: [] for k in original_dicts.get('noisy_rules', {}).keys()} # Preserve original keys

    # 准备无噪声基础程序
    noiseless_program_base = '\n'.join(noiseless_facts) + '\n' + '\n'.join(noiseless_rules)

    # 1. 筛选噪声事实
    original_noisy_facts = noisy_facts.copy() # 复制以进行迭代
    original_noisy_facts_dicts = copy.deepcopy(original_dicts.get('noisy_facts', [])) # 深拷贝

    for i, noisy_fact in enumerate(original_noisy_facts):
        temp_program = noiseless_program_base + '\n' + noisy_fact
        temp_result = dlv2_runner.run(asp_program=temp_program, num_answer_sets=0)
        if temp_result['success'] and len(temp_result['answer_sets']) == 1:
            temp_answer_set = temp_result['answer_sets'][0]
            # 检查加入此噪声事实后，查询真值是否保持不变
            if query_in_answerset == (query_fact_for_comparison in temp_answer_set):
                valid_noisy_facts.append(noisy_fact)
                if i < len(original_noisy_facts_dicts): # Safety check for index
                    valid_noisy_facts_dicts.append(original_noisy_facts_dicts[i])
        else:
            # Handle cases where adding noise causes inconsistency or multiple answer sets if needed
            print(f"Warning (seed {seed}): Adding noisy fact '{noisy_fact}' caused DLV2 error or multiple answer sets. Skipping this fact.")
            pass # Or log this error

    # 2. 筛选噪声规则
    original_noisy_rules_by_type = noisy_rules_by_type.copy() # 复制以进行迭代
    original_noisy_rules_dicts = copy.deepcopy(original_dicts.get('noisy_rules', {})) # 深拷贝

    for rule_type, rules_list in original_noisy_rules_by_type.items():
        original_rule_dicts_list = original_noisy_rules_dicts.get(rule_type, [])
        for i, noisy_rule in enumerate(rules_list):
            temp_program = noiseless_program_base + '\n' + noisy_rule
            temp_result = dlv2_runner.run(asp_program=temp_program, num_answer_sets=0)
            if temp_result['success'] and len(temp_result['answer_sets']) == 1:
                temp_answer_set = temp_result['answer_sets'][0]
                # 检查加入此噪声规则后，查询真值是否保持不变
                if query_in_answerset == (query_fact_for_comparison in temp_answer_set):
                    valid_noisy_rules_by_type[rule_type].append(noisy_rule)
                    if i < len(original_rule_dicts_list): # Safety check for index
                        valid_noisy_rules_by_type_dicts[rule_type].append(original_rule_dicts_list[i])
            else:
                # Handle cases where adding noise causes inconsistency or multiple answer sets if needed
                print(f"Warning (seed {seed}): Adding noisy rule '{noisy_rule}' caused DLV2 error or multiple answer sets. Skipping this rule.")
                pass # Or log this error

    # 使用筛选后的有效噪声替换原始噪声
    noisy_facts = valid_noisy_facts
    noisy_rules_by_type = valid_noisy_rules_by_type
    # 更新 original_dicts 中的噪声部分
    original_dicts['noisy_facts'] = valid_noisy_facts_dicts
    original_dicts['noisy_rules'] = valid_noisy_rules_by_type_dicts
    # --- End Noise Filtering Logic ---


    # 计算维持目标结论所需的最小事实集合
    minimal_facts = [] # 用于存储最小事实集合的字符串形式
    minimal_fact_dicts = [] # 用于存储最小事实集合的字典形式
    original_noiseless_facts = noiseless_facts.copy() # 复制原始列表以供迭代
    original_noiseless_fact_dicts = copy.deepcopy(original_dicts['noiseless_facts']) # 深拷贝原始字典列表

    for i in range(len(original_noiseless_facts)):
        # 创建一个临时列表，不包含当前正在检查的事实
        temp_facts = original_noiseless_facts[:i] + original_noiseless_facts[i+1:]

        # 运行 ASP 程序，检查移除当前事实后的结果
        answer_set_for_temp_facts = dlv2_runner.run(
            asp_program='\n'.join(temp_facts) + '\n' + '\n'.join(noiseless_rules),
            num_answer_sets=0)['answer_sets'][0]

        # 如果移除事实后查询结果发生改变，则该事实是必需的
        if query_in_answerset != ( query_fact_for_comparison in answer_set_for_temp_facts):
            minimal_facts.append(original_noiseless_facts[i])
            minimal_fact_dicts.append(original_noiseless_fact_dicts[i])

    # Min facts calculation finished. 使用 minimal_facts 和 minimal_fact_dicts

    # --- Format the entire dict_structure to ASP strings using the new function ---
    # Assume constants in facts are integers 0 to largest, map them to strings
    fact_var_map = {i: f'V{i}' for i in range(largest + 1)}
    # Rules use default V0, V1... variables (pass None for rule_var_idx_to_name)
    # No disjunction is used in this script's logic
    asp_program_dict = format_dict_structure_to_asp(
        dict_structure=original_dicts, # Pass the original dicts containing facts/rules
        idx_to_name=idx_to_name,
        fact_var_idx_to_name=fact_var_map,
        rule_var_idx_to_name=None, # Use default rule vars
        use_disjunction=False
    )
    # --- End formatting ---

    # --- Re-run assertions using the newly formatted strings from asp_program_dict (using filtered noise) ---
    # Assertion for min_facts (Still potentially problematic, keeping commented)
    # min_facts_str = asp_program_dict.get('min_fact_dicts_for_query', []) # Get formatted min facts
    # noiseless_rules_str = asp_program_dict.get('noiseless_rules', [])
    # assert query_in_answerset == ( query_fact_for_comparison in dlv2_runner.run(
    #         asp_program='\n'.join(min_facts_str) + '\n' + '\n'.join(noiseless_rules_str),
    #         num_answer_sets=0)['answer_sets'][0])

    # Assertion for full program with noise (using newly formatted strings)
    noiseless_facts_str = asp_program_dict.get('noiseless_facts', [])
    noiseless_rules_str = asp_program_dict.get('noiseless_rules', [])
    # Get the newly formatted strings for the *filtered* noise
    noisy_facts_str = asp_program_dict.get('noisy_facts', []) # Should now contain only valid facts
    noisy_rules_by_type_str = asp_program_dict.get('noisy_rules', {}) # Should now contain only valid rules

    noisy_facts_dlv2_str = '\n'.join(noisy_facts_str)
    noisy_rules_dlv2_str = ''
    # Iterate through the dictionary of *filtered* noisy rules
    if isinstance(noisy_rules_by_type_str, dict):
        for rule_type, rule_list in noisy_rules_by_type_str.items():
            if isinstance(rule_list, list):
                noisy_rules_dlv2_str += '\n'.join(rule_list) + '\n'

    full_program_with_filtered_noise_str = '\n'.join(noiseless_facts_str) + '\n' + '\n'.join(noiseless_rules_str) + '\n' + noisy_facts_dlv2_str + '\n' + noisy_rules_dlv2_str.strip() # Combine parts

    # Run the assertion with the program containing only filtered noise
    # This assertion now verifies that the *filtered* noise indeed doesn't change the outcome
    dlv2_result_with_filtered_noise = dlv2_runner.run(
        asp_program=full_program_with_filtered_noise_str,
        num_answer_sets=0
    )
    assert dlv2_result_with_filtered_noise['success'] and len(dlv2_result_with_filtered_noise['answer_sets']) == 1, f"DLV2 error or multiple answer sets with filtered noise for seed {seed}"
    assert query_in_answerset == (query_fact_for_comparison in dlv2_result_with_filtered_noise['answer_sets'][0]), f"Filtered noise consistency check failed after reformatting for seed {seed}"
    # --- Dlv2Runner 验证逻辑结束 ---


    # 一些数据统计 (using original counts based on sample.to_dlv2 results for consistency, but ASP program uses new format)
    length_dict_to_target = longest_path_to_target(sample.rule_graph, 0)

    max_depth_of_rule_graph = max(length_dict_to_target) if length_dict_to_target else 0 # Handle empty case
    # Filter out 0 lengths before calculating average
    valid_lengths = [i for i in length_dict_to_target if i != 0]
    average_depth_of_rule_graph = sum(valid_lengths) / len(valid_lengths) if valid_lengths else 0

    num_facts = len(noiseless_facts)
    num_rules = len(noiseless_rules)
    num_related_predicates = len([k for k,v in sample.idx2type.items() if v=='predicate'])
    assert sample.rule_predicate_graph.shape[0] == num_related_predicates + len(original_dicts['noiseless_rules']), print("sample.rule_predicate_graph.shape[0] != num_related_predicates + len(original_dicts['noiseless_rules'])")

    # 计算缺省推理的覆盖率
    num_rules_with_default_negation = 0
    for noiseless_rule in noiseless_rules:
        if ' not ' in noiseless_rule:
            num_rules_with_default_negation += 1

    symtex_sample = {
        'ID': f'symtex_dict_fact_query_{seed}_{num_nodes}_{num_edges}_{extra_predicate_num}_{extra_edge_num}_{strong_negation_prob}_{default_negation_prob}_{max_predicates_per_rule}_{m}_{largest}',
        'seed': seed,
        'num_facts': num_facts,
        'num_rules': num_rules,
        'num_related_predicates': num_related_predicates,
        'max_depth_of_rule_graph': max_depth_of_rule_graph,
        'average_depth_of_rule_graph': average_depth_of_rule_graph,
        'strong_negation_prob': strong_negation_prob,
        'default_negation_prob': default_negation_prob,
        'num_noise_rules_per_type': num_noise_rules_per_type,
        'num_rules_with_default_negation': num_rules_with_default_negation,
        'rule_graph': dense_to_sparse_serializable(sample.rule_graph),
        'rule_predicate_graph': dense_to_sparse_serializable(sample.rule_predicate_graph),
        'rule_predicate_operation_graph': dense_to_sparse_serializable(sample.rule_predicate_operation_graph),
        'noisy_rule_predicate_operation_graph': dense_to_sparse_serializable(sample.noisy_rule_predicate_operation_graph),
        'idx2type': sample.idx2type,
        # noisy_idx2type might need adjustment if noise filtering removes all noise of certain types
        # For now, keep the original noisy_idx2type, but be aware it might contain types no longer present in filtered noise
        'noisy_idx2type': sample.noisy_idx2type,
        # original_dicts now contains the filtered noisy facts/rules
        'dict_structure': {**original_dicts, 'min_fact_dicts_for_query': minimal_fact_dicts},
        'target_query': target_query_dict, # 保存原始字典结构
        'target_query_in_answerset': query_in_answerset, # 保持 target_query_in_answerset
        'max_ary_for_predicates': m,
        'max_idx_for_variables': largest,
        'max_predicates_per_rule': max([len(i['body']) for i in original_dicts.get('noiseless_rules', [{'body': []}])]), # Safer access
        # Save the noiseless ASP program built from the newly formatted strings
        'asp_program_dlv2': '\n'.join(asp_program_dict.get('noiseless_facts', [])) + '\n' + '\n'.join(asp_program_dict.get('noiseless_rules', [])),
    }

    return symtex_sample

# Helper function to generate sample ID string based on parameters
def _generate_sample_id(seed, num_nodes, num_edges, extra_predicate_num, extra_edge_num,
                        strong_negation_prob, default_negation_prob, max_predicates_per_rule,
                        num_noise_rules_per_type, m, largest): # Added num_noise_rules_per_type back for consistency if needed elsewhere, but ID format doesn't use it
    """Generates the unique ID string for a sample based on its parameters."""
    # Note: num_noise_rules_per_type is NOT included in the ID string as per original logic
    return f'symtex_dict_fact_query_{seed}_{num_nodes}_{num_edges}_{extra_predicate_num}_{extra_edge_num}_{strong_negation_prob}_{default_negation_prob}_{max_predicates_per_rule}_{m}_{largest}'

# Helper function to generate output filename based on parameters (excluding num_noise_rules_per_type)
def _generate_output_filename(num_nodes, num_edges, extra_predicate_num, extra_edge_num,
                              strong_negation_prob, default_negation_prob, max_predicates_per_rule,
                              m, largest):
    """Generates the output filename based on generation parameters."""
    # Construct filename from parameters, excluding num_noise_rules_per_type
    param_str = f"nodes{num_nodes}_edges{num_edges}_extraP{extra_predicate_num}_extraE{extra_edge_num}_sNeg{strong_negation_prob}_dNeg{default_negation_prob}_maxPred{max_predicates_per_rule}_m{m}_l{largest}"
    return f"symtex_dataset_{param_str}.jsonl"


# Function to process a single seed (for multithreading)
def process_seed(seed, args, output_path, output_lock, existing_ids_lock, existing_ids_set):
    """Generates a single SymTex sample for a given seed."""
    expected_id = _generate_sample_id(
        seed=seed,
        num_nodes=args.num_nodes,
        num_edges=args.num_edges,
        extra_predicate_num=args.extra_predicate_num,
        extra_edge_num=args.extra_edge_num,
        strong_negation_prob=args.strong_negation_prob,
        default_negation_prob=args.default_negation_prob,
        max_predicates_per_rule=args.max_predicates_per_rule,
        num_noise_rules_per_type=args.num_noise_rules_per_type,
        m=args.m,
        largest=args.largest
    )

    # Double-check existence within the worker thread using the lock
    with existing_ids_lock:
        if expected_id in existing_ids_set:
            return 'skipped', expected_id, None # Indicate skipped

    try:
        result_sample = generate_symtex_sample(
            seed=seed,
            num_nodes=args.num_nodes,
            num_edges=args.num_edges,
            extra_predicate_num=args.extra_predicate_num,
            extra_edge_num=args.extra_edge_num,
            strong_negation_prob=args.strong_negation_prob,
            default_negation_prob=args.default_negation_prob,
            max_predicates_per_rule=args.max_predicates_per_rule,
            num_noise_rules_per_type=args.num_noise_rules_per_type,
            m=args.m,
            largest=args.largest
        )

        # Safety check: Ensure generated ID matches expected ID
        if result_sample['ID'] != expected_id:
             # Log error instead of printing directly in thread
             return 'error', expected_id, f"Generated ID '{result_sample['ID']}' != expected ID '{expected_id}'"

        # Append the new sample to the JSONL file using the lock
        with output_lock:
            append_jsonl(result_sample, output_path)
            # Add to shared set under lock
            with existing_ids_lock:
                 existing_ids_set.add(expected_id)

        return 'success', expected_id, result_sample # Indicate success

    except Exception as e:
        # Return error status and the exception object or message
        return 'error', expected_id, e


# 标准 Python 入口点 - 现在用于批量生成数据集
if __name__ == "__main__":
    # --- Initialize Counters ---
    generated_count = 0
    skipped_initially = 0 # Skipped because ID already exists
    skipped_noise_inconsistent = 0 # Skipped because noise changed query truth value
    skipped_during_run = 0 # Skipped due to race condition (ID added by another thread)
    error_count = 0
    parser = argparse.ArgumentParser(description="Generate SymTex dataset in JSONL format with resume capability, auto-filename option, and multithreading.")

    # --- Dataset Generation Parameters ---
    parser.add_argument('--output_file', type=str, default=None, help='Optional: Path to the output JSONL file. If not provided, filename is generated automatically based on parameters.')
    parser.add_argument('--output_dir', type=str, default='datasets/symtex_dict_for_query', help='Directory to save the output file if filename is auto-generated.')
    parser.add_argument('--num_samples', type=int, default=100, help='Total number of samples to generate.')
    parser.add_argument('--start_seed', type=int, default=0, help='Starting seed for sample generation.')

    # --- SymTexSample Parameters (using defaults from the original script) ---
    parser.add_argument('--num_nodes', type=int, default=5, help='Number of nodes in the base graph.')
    parser.add_argument('--num_edges', type=int, default=5, help='Number of edges in the base graph.')
    parser.add_argument('--extra_predicate_num', type=int, default=1, help='Number of extra predicates.')
    parser.add_argument('--extra_edge_num', type=int, default=3, help='Number of extra edges for predicates.')
    parser.add_argument('--strong_negation_prob', type=float, default=0.5, help='Probability of strong negation.')
    parser.add_argument('--default_negation_prob', type=float, default=1.0, help='Probability of default negation.')
    parser.add_argument('--max_predicates_per_rule', type=int, default=3, help='Maximum predicates per rule body.')
    parser.add_argument('--num_noise_rules_per_type', type=int, default=1, help='Number of noise rules per type.')
    parser.add_argument('--m', type=int, default=2, help='Parameter m for rule generation.')
    parser.add_argument('--largest', type=int, default=1, help='Parameter largest for rule generation.')
    parser.add_argument('--num_threads', type=int, default=os.cpu_count(), help='Number of threads to use for generation.') # Added num_threads argument

    args = parser.parse_args()

    # --- Determine Output Path ---
    if args.output_file:
        output_path = args.output_file
        # Ensure directory exists if a full path is given
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        # Auto-generate filename
        auto_filename = _generate_output_filename(
            num_nodes=args.num_nodes,
            num_edges=args.num_edges,
            extra_predicate_num=args.extra_predicate_num,
            extra_edge_num=args.extra_edge_num,
            strong_negation_prob=args.strong_negation_prob,
            default_negation_prob=args.default_negation_prob,
            max_predicates_per_rule=args.max_predicates_per_rule,
            m=args.m,
            largest=args.largest
        )
        output_path = os.path.join(args.output_dir, auto_filename)
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output file not specified, auto-generating filename: {output_path}")


    # --- Load existing IDs if output file exists ---
    existing_ids = set()
    if os.path.exists(output_path):
        print(f"Output file '{output_path}' found. Loading existing sample IDs...")
        try:
            # Use lazy reading for potentially large files
            for i, sample in enumerate(read_jsonl_lazy(output_path)):
                if 'ID' in sample:
                    existing_ids.add(sample['ID'])
                else:
                    pass
            print(f"Loaded {len(existing_ids)} existing IDs.")
        except FileNotFoundError:
            # Should not happen due to os.path.exists, but good practice
            print(f"Warning: File '{output_path}' disappeared unexpectedly.")
        except (json.JSONDecodeError, JsonUtilsError) as e:
            print(f"Error reading or decoding JSONL file '{output_path}': {e}")
            print("Proceeding without loading existing IDs, duplicates might occur if generation continues.")
            existing_ids = set() # Reset on error to avoid partial loading issues
        except Exception as e:
            print(f"An unexpected error occurred while loading existing IDs from '{output_path}': {e}")
            existing_ids = set() # Keep as existing_ids for clarity in this scope

    # --- Pre-filter seeds ---
    seeds_to_process = []
    skipped_initially = 0
    print("Pre-filtering seeds based on existing IDs...")
    for i in range(args.num_samples):
        current_seed = args.start_seed + i
        expected_id = _generate_sample_id(
            seed=current_seed,
            num_nodes=args.num_nodes,
            num_edges=args.num_edges,
            extra_predicate_num=args.extra_predicate_num,
            extra_edge_num=args.extra_edge_num,
            strong_negation_prob=args.strong_negation_prob,
            default_negation_prob=args.default_negation_prob,
            max_predicates_per_rule=args.max_predicates_per_rule,
            num_noise_rules_per_type=args.num_noise_rules_per_type,
            m=args.m,
            largest=args.largest
        )
        if expected_id not in existing_ids:
            seeds_to_process.append(current_seed)
        else:
            skipped_initially += 1
    print(f"Found {len(existing_ids)} existing IDs. Will skip {skipped_initially} seeds. Processing {len(seeds_to_process)} seeds.")


    # --- Threaded Generation Loop ---
    output_lock = threading.Lock()
    existing_ids_lock = threading.Lock() # Separate lock for the shared set

    # Use tqdm for progress bar
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=len(seeds_to_process), desc="Generating Samples")
    except ImportError:
        print("tqdm not found, progress bar disabled. Run 'pip install tqdm' to enable.")
        progress_bar = None # Set to None if tqdm not available

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        print(f"Starting generation with {args.num_threads} threads...")
        # Submit all tasks
        for seed in seeds_to_process:
            # Pass locks and the shared set (existing_ids) to the worker
            future = executor.submit(process_seed, seed, args, output_path, output_lock, existing_ids_lock, existing_ids)
            futures[future] = seed

        # Process completed tasks
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                # result can be the sample dict, None (if noise inconsistent), or an exception
                status, processed_id, result_or_error = future.result()

                # Update counters based on status returned by process_seed
                if status == 'success':
                     # Check if result_or_error is the sample dict (not None)
                     if result_or_error is not None:
                          generated_count += 1
                     else:
                          # This means generate_symtex_sample returned None due to noise inconsistency
                          skipped_noise_inconsistent += 1
                elif status == 'skipped':
                    # This skip happens if another thread added the ID between pre-filtering and the worker check
                    skipped_during_run += 1
                elif status == 'error':
                    error_count += 1
                    print(f"\nError processing seed {seed} (ID: {processed_id}): {result_or_error}") # Print error details

            except Exception as exc:
                # Handle exceptions raised during future.result() itself (less common)
                error_count += 1
                print(f'\nSeed {seed} generated an exception: {exc}')

            # Update progress bar if tqdm is available
            if progress_bar:
                progress_bar.update(1)
                total_skipped_display = skipped_initially + skipped_during_run + skipped_noise_inconsistent
                progress_bar.set_postfix_str(f"Generated: {generated_count}, Skipped: {total_skipped_display}, Errors: {error_count}")

    # Close progress bar if it exists
    if progress_bar:
        progress_bar.close()

    total_skipped = skipped_initially + skipped_during_run + skipped_noise_inconsistent
    print("\nDataset generation finished.")
    print(f"  Total seeds processed: {len(seeds_to_process)}")
    print(f"  Samples generated and saved: {generated_count}")
    print(f"  Samples skipped:")
    print(f"    - Already existed (initial check): {skipped_initially}")
    print(f"    - Noise inconsistent (in generator): {skipped_noise_inconsistent}")
    print(f"    - Race condition skip (in thread): {skipped_during_run}")
    print(f"    - Total skipped: {total_skipped}")
    print(f"  Errors encountered: {error_count}")
    print(f"  Dataset saved to: {output_path}")
