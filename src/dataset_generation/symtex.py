import json
import random
from typing import Dict, Optional, Set, List, Tuple, FrozenSet # Added FrozenSet
from collections import defaultdict # Added for connectivity check
import itertools # Added for combinations in rule modification
import copy # Added for deep copying rule dicts

import numpy as np

# Added imports for DLV2 check
from src.utils.dlv2_runner import Dlv2Runner, Dlv2RunnerError

# Updated import
from src.dataset_generation.sample_generation import (
    generate_strongly_connected_dag, expand_graph, apply_negations,
    STRONG_NEGATION_VALUE, DEFAULT_NEGATION_VALUE,
    get_sorted_variable_list, get_modified_noise_edge_flag, # Added imports
    generate_asp_dict_from_graph # Import the new function
)
from src.dataset_generation.variable_generation import solve_wrapper


# --- Helper Functions for DLV2 Conversion (Moved outside the class) ---

def _format_atom_dlv2(pred_dict: dict, idx_to_name: Dict[int, str], is_fact: bool = False, is_body: bool = False) -> str:
    """Formats a predicate dictionary into a DLV2 atom string."""
    idx = pred_dict['predicateIdx']
    name = idx_to_name.get(idx, f"P{idx}") # Fallback name
    variables = pred_dict['variables']
    strong_neg = pred_dict.get('strong negation', False)
    default_neg = pred_dict.get('default negation', False) and is_body # Default negation only in body

    prefix = ""
    if default_neg:
        prefix += "not "
    if strong_neg:
        prefix += "- " # Add space after - for clarity if not default neg
        if default_neg: prefix = "not -" # Combine if both

    # Standardize variables and apply quotes for facts
    formatted_vars = []
    for v in variables:
        # Ensure variable is in "V<number>" format string
        var_str_base = f"V{v}" if isinstance(v, int) else str(v)
        # Apply quotes only for facts
        if is_fact:
            formatted_vars.append(f'"{var_str_base}"')
        else:
            formatted_vars.append(var_str_base)
    var_str = ", ".join(formatted_vars)
    if var_str:
        return f"{prefix}{name}({var_str})"
    else:
        return f"{prefix}{name}"

def _format_rule_dict_to_dlv2(rule_dict: dict, idx_to_name: Dict[int, str], use_disjunction: bool) -> List[str]:
    """Formats a rule dictionary into one or more DLV2 rule strings."""
    formatted_rules = []
    head_atoms = [_format_atom_dlv2(p, idx_to_name, is_fact=False, is_body=False) for p in rule_dict.get('head', [])]
    body_atoms = [_format_atom_dlv2(p, idx_to_name, is_fact=False, is_body=True) for p in rule_dict.get('body', [])]
    body_str = ", ".join(body_atoms)

    if not head_atoms: return [] # Skip rule if no head

    if use_disjunction or len(head_atoms) == 1:
        head_str = " | ".join(head_atoms)
        rule_str = f"{head_str} :- {body_str}." if body_str else f"{head_str}."
        formatted_rules.append(rule_str)
    else:
        # Split into multiple rules
        for head_atom in head_atoms:
            rule_str = f"{head_atom} :- {body_str}." if body_str else f"{head_atom}."
            formatted_rules.append(rule_str)
    return formatted_rules

def _validate_and_modify_rule_for_safety(
    rule_dict: dict,
    dlv2_runner: Optional[Dlv2Runner],
    idx_to_name: Dict[int, str],
    use_disjunction: bool,
    rule_context_info: str
) -> dict:
    """
    Checks a rule dictionary for DLV2 safety errors and attempts modifications.

    :param rule_dict: The rule dictionary to validate.
    :param dlv2_runner: The Dlv2Runner instance (can be None).
    :param idx_to_name: Mapping from predicate index to name.
    :param use_disjunction: Flag for formatting rules.
    :param rule_context_info: String for logging context (e.g., "noiseless rule idx 5").
    :return: The validated (potentially modified) rule dictionary.
    """
    validated_rule_dict = rule_dict # Start with the original

    if not dlv2_runner:
        return validated_rule_dict # Skip checks if runner is not available

    rule_modified = False
    initial_dlv2_rules = _format_rule_dict_to_dlv2(rule_dict, idx_to_name, use_disjunction)
    initial_check_failed = False
    safety_error_found = False

    # Initial check for all generated strings from the dict
    for rule_str in initial_dlv2_rules:
        result = dlv2_runner.run(rule_str)
        if not result['success']:
            initial_check_failed = True
            if "Safety Error" in (result['error_message'] or ""):
                safety_error_found = True
                # print(f"    Safety Error found in {rule_context_info}: {rule_str}")
                break # Stop checking this rule dict if safety error found
            else:
                # print(f"    Other Error found in {rule_context_info}: {rule_str} -> {result['error_message']}")
                # Decide if we should break or continue checking other parts of a multi-head rule
                break # Break on any error for now

    if initial_check_failed and safety_error_found:
        # print(f"    Attempting modification for {rule_context_info}...")
        # Identify candidates for modification (body predicates with default negation)
        candidates = []
        for i, pred_dict in enumerate(rule_dict.get('body', [])):
            if pred_dict.get('default negation', False):
                candidates.append(i) # Store index

        if not candidates:
            print(f"      No modification candidates (default negation in body) found for {rule_context_info}.")
            pass # Keep original rule if no candidates
        else:
            modification_successful = False
            # Strategy 1: Try flipping one candidate
            # print(f"      Modification candidates (indices): {candidates}")
            for candidate_idx in candidates:
                modified_rule_dict_attempt = rule_dict
                # Flip negation flags for the candidate
                pred_to_modify = modified_rule_dict_attempt['body'][candidate_idx]
                pred_to_modify['default negation'] = not pred_to_modify.get('default negation', False)
                pred_to_modify['strong negation'] = not pred_to_modify.get('strong negation', False)
                # print(f"      Trying modification (Strategy 1): Flipping predicate at index {candidate_idx}")

                modified_dlv2_rules_attempt = _format_rule_dict_to_dlv2(modified_rule_dict_attempt, idx_to_name, use_disjunction)
                modification_check_passed = True
                for rule_str_mod in modified_dlv2_rules_attempt:
                    result_mod = dlv2_runner.run(rule_str_mod)
                    if not result_mod['success']:
                        modification_check_passed = False
                        # print(f"        Modification failed check: {rule_str_mod} -> {result_mod['error_message']}")
                        break # Stop checking this modification attempt

                if modification_check_passed:
                    # print(f"      Modification successful (Strategy 1 - flipped index {candidate_idx})!")
                    validated_rule_dict = modified_rule_dict_attempt
                    rule_modified = True
                    modification_successful = True
                    break # Exit strategy 1 loop

            # Strategy 2: Try flipping two candidates (if strategy 1 failed and >= 2 candidates)
            if not modification_successful and len(candidates) >= 2:
                # print("      Strategy 1 failed. Trying Strategy 2 (flipping pairs)...")
                for combo in itertools.combinations(candidates, 2):
                    idx1, idx2 = combo
                    modified_rule_dict_attempt = copy.deepcopy(rule_dict)
                    # Flip negation flags for the pair
                    pred1 = modified_rule_dict_attempt['body'][idx1]
                    pred1['default negation'] = not pred1.get('default negation', False)
                    pred1['strong negation'] = not pred1.get('strong negation', False)
                    pred2 = modified_rule_dict_attempt['body'][idx2]
                    pred2['default negation'] = not pred2.get('default negation', False)
                    pred2['strong negation'] = not pred2.get('strong negation', False)
                    # print(f"      Trying modification (Strategy 2): Flipping predicates at indices {idx1} and {idx2}")

                    modified_dlv2_rules_attempt = _format_rule_dict_to_dlv2(modified_rule_dict_attempt, idx_to_name, use_disjunction)
                    modification_check_passed = True
                    for rule_str_mod in modified_dlv2_rules_attempt:
                        result_mod = dlv2_runner.run(rule_str_mod)
                        if not result_mod['success']:
                            modification_check_passed = False
                            # print(f"        Modification failed check: {rule_str_mod} -> {result_mod['error_message']}")
                            break

                    if modification_check_passed:
                        # print(f"      Modification successful (Strategy 2 - flipped indices {idx1}, {idx2})!")
                        validated_rule_dict = modified_rule_dict_attempt
                        rule_modified = True
                        modification_successful = True
                        break # Exit strategy 2 loop

            if not modification_successful:
                 print(f"      Modification attempts failed for {rule_context_info}. Keeping original.")
                 # validated_rule_dict remains rule_dict

    return validated_rule_dict


class SymTexSample:
    """
    用于保存一个 SymTex 样本的基本信息。
    """
    def __init__(self,
        num_nodes: int,
        num_edges: int,
        seed=None,
        extra_predicate_num: int = 0,
        extra_edge_num: int = 0,
        strong_negation_prob: float = 0.0,
        default_negation_prob: float = 0.0,
        max_predicates_per_rule: int = 3,
        num_noise_rules_per_type: int = 0,
        ): # Added max_predicates_per_rule
        """
        初始化 SymTexSample 实例。

        :param num_nodes: 初始 Rule 节点数
        :param num_edges: 初始 Rule 节点间的边数
        :param seed: 随机种子，用于复现生成过程 (可选)
        :param extra_predicate_num: 额外添加的 Predicate 节点数量 (默认为 0)
        :param extra_edge_num: 额外添加的 Predicate -> Rule 边数量 (默认为 0)
        :param strong_negation_prob: 对 Predicate 应用强否定的概率 (默认为 0.0)
        :param default_negation_prob: 对 Predicate 应用默认否定的概率 (默认为 0.0)
        :param max_predicates_per_rule: 噪声规则中体部谓词的最大数量 (默认为 3)
        """
        self.rules = None
        self.facts = None
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.seed = seed
        self.extra_predicate_num = extra_predicate_num
        self.extra_edge_num = extra_edge_num
        self.strong_negation_prob = strong_negation_prob
        self.default_negation_prob = default_negation_prob
        self.max_predicates_per_rule = max_predicates_per_rule # Store the new parameter
        # 可以在这里添加文档字符串或注释，说明这个类的目的
        # 例如，可以后续添加存储生成结果的属性
        self.rule_graph = None
        self.rule_predicate_graph = None
        self.rule_predicate_operation_graph = None
        self.idx2type = None
        self.num_predicates = None
        self.predicates = None
        self.predicate_variables = {} # Initialize predicate variables storage
        self.noise_rule_info: Dict[int, str] = {} # Added to store noise rule type mapping
        self.noisy_rule_predicate_operation_graph: Optional[np.ndarray] = None # For noisy graph
        self.noisy_idx2type: Optional[Dict[int, str]] = None # For noisy type map
        self.num_noise_rules_per_type = num_noise_rules_per_type
        self.noise_rules = None # Initialize noisy rules storage
        self.noise_facts = None # Initialize noisy facts storage

        self._sample_generate()
        # Optionally call add_noise here if it should always run after generation
        # Or call it manually after creating the SymTexSample instance.
        self.add_noise(num_noise_rules_per_type=num_noise_rules_per_type, seed=self.seed) # Example if called here

    def _sample_generate(self):
        """
        生成样本的规则图和规则-谓词图。

        该方法会调用generate_strongly_connected_dag生成规则图，然后调用expand_graph
        生成规则-谓词图，并将结果保存到类的属性中。
        """
        # 生成规则图
        self.rule_graph = generate_strongly_connected_dag(
            self.num_nodes,
            self.num_edges,
            seed=self.seed
        )

        # 生成规则-谓词图 (传入 max_predicates_per_rule)
        self.rule_predicate_graph, self.idx2type = expand_graph(
            self.rule_graph,
            extra_predicate_num=self.extra_predicate_num,
            extra_edge_num=self.extra_edge_num,
            max_predicates_per_rule=self.max_predicates_per_rule, # Pass the constraint
            seed=self.seed
        )

        self.rule_predicate_operation_graph = apply_negations(
            self.rule_predicate_graph,
            self.idx2type,
            strong_negation_prob=self.strong_negation_prob,
            default_negation_prob=self.default_negation_prob,
            seed=self.seed
        )

        # 添加断言以验证生成的图是否符合参数设置
        assert self.rule_graph is not None, "规则图未生成"
        assert self.rule_predicate_graph is not None, "规则-谓词图未生成"
        assert self.idx2type is not None, "节点类型映射未生成"
        assert self.rule_predicate_operation_graph is not None, "规则-谓词操作图未生成"

        # 验证规则图的边数是否符合期望
        actual_num_edges = np.sum(self.rule_graph)
        assert actual_num_edges == self.num_edges, f"生成的规则图边数 ({actual_num_edges}) 与期望边数 ({self.num_edges}) 不符"

        # 验证规则-谓词图的节点扩展是否生效
        assert len(self.idx2type) == self.rule_predicate_graph.shape[0], f"规则-谓词图的节点数 ({len(self.idx2type)}) 与期望节点数 ({self.rule_predicate_graph.shape[0]}) 不符"

        # 验证规则-谓词图的边扩展是否生效
        actual_num_predicate_edges = np.sum(self.rule_predicate_graph)
        num_leaf_nodes = sum(np.sum(self.rule_graph, axis=0) == 0)
        num_expected_edges = num_leaf_nodes + self.num_edges*2 + 1 + self.extra_edge_num
        assert actual_num_predicate_edges == num_expected_edges, f"生成的规则-谓词图边数 ({actual_num_predicate_edges}) 与期望边数 ({num_expected_edges}) 不符"

        self.num_predicates = len([k for k,v in self.idx2type.items() if v == 'predicate'])
        self.num_rules = len([k for k,v in self.idx2type.items() if v == 'rule'])
        # self.predicate_variables = {} # Moved initialization to __init__

    def add_noise(self, num_noise_rules_per_type: int, seed: Optional[int] = None):
        """
        在现有的 rule_predicate_operation_graph 中添加五种类型的噪声规则。

        :param num_noise_rules_per_type: 每种类型要添加的噪声规则数量。
        :param seed: 随机种子，用于复现噪声生成过程 (可选)。
        """
        if num_noise_rules_per_type <= 0:
            return  # No noise to add

        if self.rule_predicate_operation_graph is None or self.idx2type is None:
            raise ValueError("图数据尚未生成，无法添加噪声。请先调用 _sample_generate()")

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Start with copies of the original graph and type map
        graph_copy = self.rule_predicate_operation_graph.copy()
        idx2type_copy = self.idx2type.copy()
        current_num_nodes = graph_copy.shape[0]
        next_node_idx = current_num_nodes

        # --- 1. Initialization ---
        # Use the copied idx2type
        all_original_rule_indices = {idx for idx, type_ in self.idx2type.items() if type_ == 'rule'} # Use original idx2type here
        original_graph = self.rule_predicate_operation_graph # Reference original graph

        existing_pred_indices = {idx for idx, type_ in idx2type_copy.items() if type_ == 'predicate'}
        if not existing_pred_indices:
            print("Warning: No existing predicates found. Cannot add noise effectively.")
            return  # Cannot add noise without predicates

        # Identify original fact predicates (no incoming edge from original rules in original graph)
        original_fact_pred_indices = set()
        for p_idx in existing_pred_indices:
             # Check only indices within the bounds of the original graph
             if p_idx < original_graph.shape[0]:
                 is_fact = True
                 for r_idx in all_original_rule_indices:
                     if r_idx < original_graph.shape[0] and original_graph[r_idx, p_idx] > 0:
                         is_fact = False
                         break
                 if is_fact:
                     original_fact_pred_indices.add(p_idx)

        # Store the INITIAL predicate flags (before noise modification)
        initial_predicate_flags: Dict[int, int] = {}
        # Find the target predicate (assuming rule 0 is the target rule)
        target_rule_idx = 0
        p_target_idx: Optional[int] = None
        # Use the copied graph to find target
        for p_idx in existing_pred_indices:
            if graph_copy[target_rule_idx, p_idx] > 0:
                p_target_idx = p_idx
                break
        if p_target_idx is None:
            print(
                f"Warning: Could not find target predicate connected from rule {target_rule_idx}. Cannot add noise types a & b.")
            # Decide if we should proceed with other types or stop
            # For now, let's allow adding other types if p_target_idx is None

        # Determine INITIAL edge flags for existing predicates
        for p_idx in existing_pred_indices:
            flag = 1  # Default to positive if isolated
            # Check outgoing edges first using the ORIGINAL graph to get the initial state
            # Use original_graph and original node count (current_num_nodes)
            found_flag = False
            if p_idx < current_num_nodes: # Check bounds for original graph
                for target_idx in range(current_num_nodes):
                    if original_graph[p_idx, target_idx] > 0:
                        flag = original_graph[p_idx, target_idx]
                        found_flag = True
                        break
                # If no outgoing, check incoming edges in original graph
                if not found_flag:
                    for source_idx in range(current_num_nodes):
                        if original_graph[source_idx, p_idx] > 0:
                            flag = original_graph[source_idx, p_idx]
                            found_flag = True
                            break
            # Store the determined initial flag
            initial_predicate_flags[p_idx] = flag | 1 # Ensure base 1 is always present

        newly_created_pred_indices: Set[int] = set()
        new_edges_to_add: List[Tuple[int, int, int]] = []
        generated_rule_signatures: Set[
            Tuple[frozenset, int]] = set()  # Store (body_preds_frozenset, head_pred) to avoid duplicates

        # --- Helper function to create a new predicate ---
        def create_new_predicate():
            nonlocal next_node_idx
            p_new_idx = next_node_idx
            idx2type_copy[p_new_idx] = 'predicate' # Modify the copy
            newly_created_pred_indices.add(p_new_idx)
            # Determine negation for the new predicate
            neg_val = 0
            if random.random() < self.strong_negation_prob:
                neg_val |= STRONG_NEGATION_VALUE
            if random.random() < self.default_negation_prob:
                neg_val |= DEFAULT_NEGATION_VALUE
            initial_flag = 1 | neg_val # Calculate initial flag
            initial_predicate_flags[p_new_idx] = initial_flag # Store initial flag
            next_node_idx += 1
            return p_new_idx

        # --- 2. Loop through noise types ---
        noise_types = ['a', 'b', 'c', 'd', 'e']
        for noise_type in noise_types:
            added_count = 0
            attempts = 0
            max_attempts = num_noise_rules_per_type * 10  # Allow for some retries due to duplicates or lack of options

            # Pre-filter predicate pools for efficiency, excluding original facts from body pools
            existing_preds_no_target_no_facts = list(
                 (existing_pred_indices - {p_target_idx} if p_target_idx is not None else existing_pred_indices) - original_fact_pred_indices
            )
            existing_preds_no_target = list( # Pool for heads (can include facts)
                 existing_pred_indices - {p_target_idx} if p_target_idx is not None else existing_pred_indices
            )

            # Check if enough non-fact predicates exist for body generation
            if not existing_preds_no_target_no_facts and noise_type in ['a', 'c', 'd']:
                 # print(f"Warning: Not enough existing non-fact predicates (excluding target) to generate noise type {noise_type} body.")
                 continue # Skip this type if no suitable predicates exist for the body

            # Check if enough predicates exist for head generation (can include facts)
            if not existing_preds_no_target and noise_type in ['b', 'c', 'd']:
                # print(f"Warning: Not enough existing predicates (excluding target) to generate noise type {noise_type} head.")
                continue  # Skip this type if no suitable predicates exist

            while added_count < num_noise_rules_per_type and attempts < max_attempts:
                attempts += 1

                # Create new rule node
                r_noise_idx = next_node_idx
                idx2type_copy[r_noise_idx] = 'rule' # Modify the copy
                next_node_idx += 1

                num_body = random.randint(1, self.max_predicates_per_rule)
                current_rule_body_indices = set()
                current_rule_head_idx = -1  # Placeholder

                try:  # Use try-except to handle potential sampling issues (e.g., not enough predicates)
                    # --- Select Head and Body based on type ---
                    if noise_type == 'a':
                        if p_target_idx is None: continue  # Cannot generate type a
                        current_rule_head_idx = p_target_idx
                        # Body: Choose from existing (excluding target AND excluding original facts)
                        if not existing_preds_no_target_no_facts: raise ValueError(
                            "Not enough existing non-fact predicates for type a body")
                        k = min(num_body, len(existing_preds_no_target_no_facts))
                        current_rule_body_indices = set(random.sample(existing_preds_no_target_no_facts, k))

                    elif noise_type == 'b':
                        if p_target_idx is None: continue  # Cannot generate type b
                        # Head: Choose from existing (excluding target, can be fact)
                        if not existing_preds_no_target: raise ValueError(
                            "Not enough existing predicates for type b head")
                        current_rule_head_idx = random.choice(existing_preds_no_target)
                        # Body: Must include target, others from existing (excluding target AND excluding original facts)
                        current_rule_body_indices.add(p_target_idx)
                        if num_body > 1 and existing_preds_no_target_no_facts:
                             k = min(num_body - 1, len(existing_preds_no_target_no_facts))
                             current_rule_body_indices.update(random.sample(existing_preds_no_target_no_facts, k))

                    elif noise_type == 'c':
                        # Head: Choose from existing (excluding target, can be fact)
                        if not existing_preds_no_target: raise ValueError("Not enough existing predicates for type c head")
                        current_rule_head_idx = random.choice(existing_preds_no_target)
                        # Body: Choose from existing (excluding target AND excluding original facts)
                        if not existing_preds_no_target_no_facts: raise ValueError("Not enough existing non-fact predicates for type c body")
                        k = min(num_body, len(existing_preds_no_target_no_facts))
                        # Ensure body doesn't contain the head predicate
                        body_pool_c = [p for p in existing_preds_no_target_no_facts if p != current_rule_head_idx]
                        if len(body_pool_c) < k: k = len(body_pool_c) # Adjust k if pool is smaller
                        if k > 0:
                             current_rule_body_indices = set(random.sample(body_pool_c, k))
                        else: # Handle case where head was the only non-fact predicate
                             continue # Cannot form a valid body

                        # Original check for self-dependency is implicitly handled by using body_pool_c
                        # if current_rule_head_idx in current_rule_body_indices:
                        #     # Attempt to re-sample body or skip this attempt
                        #     if len(existing_preds_no_target_no_facts) > k:
                        #         current_rule_body_indices = set(
                        #             random.sample([p for p in existing_preds_no_target_no_facts if p != current_rule_head_idx], k)
                        #         )
                        #     else:
                        #         continue  # Skip if cannot avoid self-dependency

                    elif noise_type == 'd':
                        # Head & Body: Mix of existing (no target) and new
                        num_new = random.randint(1, num_body + 1)  # Number of new predicates to create
                        current_new_preds = {create_new_predicate() for _ in range(num_new)}

                        # Head: Choose from existing (no target, can be fact) OR new
                        head_pool = existing_preds_no_target + list(current_new_preds) # Use pool that includes facts
                        if not head_pool: raise ValueError("No predicates available for type d head")
                        current_rule_head_idx = random.choice(head_pool)

                        # Body: Mix of existing (no target, no facts) and new, ensuring at least one of each if possible
                        body_pool_existing_no_facts = [p for p in existing_preds_no_target_no_facts if
                                                       p != current_rule_head_idx]  # Exclude head if it was existing non-fact
                        body_pool_new = [p for p in current_new_preds if
                                         p != current_rule_head_idx]  # Exclude head if it was new

                        # Select body predicates, prioritizing mix
                        if num_body == 1:
                             # Choose randomly between existing non-fact or new
                             pool = ([p for p in body_pool_existing_no_facts] if random.random() < 0.5 else []) + \
                                    ([p for p in body_pool_new] if body_pool_new else [])
                             if not pool: pool = body_pool_existing_no_facts + body_pool_new # Fallback
                             if not pool: raise ValueError("No predicates available for type d body (n=1)")
                             current_rule_body_indices = {random.choice(pool)}
                        elif num_body >= 2:
                             # Ensure at least one from each pool if available
                             if body_pool_existing_no_facts: current_rule_body_indices.add(random.choice(body_pool_existing_no_facts))
                             if body_pool_new: current_rule_body_indices.add(random.choice(body_pool_new))
                             # Fill remaining spots
                             remaining_needed = num_body - len(current_rule_body_indices)
                             combined_pool = [p for p in body_pool_existing_no_facts + body_pool_new if
                                              p not in current_rule_body_indices]
                             k = min(remaining_needed, len(combined_pool))
                             if k > 0:
                                 current_rule_body_indices.update(random.sample(combined_pool, k))
                        if not current_rule_body_indices: raise ValueError("Failed to select body for type d")


                    elif noise_type == 'e':
                        # Head & Body: All new predicates
                        num_new = num_body + 1
                        current_new_preds = {create_new_predicate() for _ in range(num_new)}
                        new_preds_list = list(current_new_preds)
                        current_rule_head_idx = new_preds_list[0]
                        current_rule_body_indices = set(new_preds_list[1:])

                    # --- Check for duplicates and add edges ---
                    if not current_rule_body_indices or current_rule_head_idx == -1:
                        # Failed to generate valid rule, decrement rule node index
                        del idx2type_copy[r_noise_idx] # Modify the copy
                        next_node_idx -= 1
                        continue  # Try again

                    rule_sig = (frozenset(current_rule_body_indices), current_rule_head_idx)
                    if rule_sig in generated_rule_signatures:
                        # Duplicate rule structure: remove rule node if it exists, then decrement index
                        node_was_added = False
                        if r_noise_idx in idx2type_copy:
                            del idx2type_copy[r_noise_idx]
                            node_was_added = True # Mark that a node was actually added and now removed

                        if node_was_added:
                            next_node_idx -= 1 # Decrement only if a node was truly added and removed

                        # Also potentially remove newly created predicates if they are now unused (complex)
                        continue  # Try again

                    generated_rule_signatures.add(rule_sig)

                    # Record the noise type for this rule
                    self.noise_rule_info[r_noise_idx] = noise_type # Store noise type mapping

                    # Add edges for the rule using MODIFIED flags based on initial state
                    head_initial_flag = initial_predicate_flags.get(current_rule_head_idx, 1) # Default to 1 if somehow missing
                    # Use imported function
                    modified_head_flag = get_modified_noise_edge_flag(head_initial_flag)
                    new_edges_to_add.append(
                        (r_noise_idx, current_rule_head_idx, modified_head_flag))

                    for p_body_idx in current_rule_body_indices:
                        body_initial_flag = initial_predicate_flags.get(p_body_idx, 1) # Default to 1 if somehow missing
                        # Use imported function
                        modified_body_flag = get_modified_noise_edge_flag(body_initial_flag)
                        new_edges_to_add.append((p_body_idx, r_noise_idx, modified_body_flag))

                    added_count += 1

                except (ValueError, IndexError) as e:
                    # Exception occurred: remove rule node if it exists, then decrement index
                    node_was_added_exception = False
                    if r_noise_idx in idx2type_copy:
                        del idx2type_copy[r_noise_idx]
                        node_was_added_exception = True # Mark that a node was added and now removed

                    if node_was_added_exception:
                        next_node_idx -= 1 # Decrement only if a node was truly added and removed

                    # Potentially clean up newly created predicates if unused (complex)
                    continue  # Try again or move to next type if max_attempts reached

        # --- 3. Update Graph ---
        final_num_nodes = next_node_idx
        if final_num_nodes > current_num_nodes:
            # Resize graph matrix using the copy as base
            new_graph = np.zeros((final_num_nodes, final_num_nodes), dtype=int)
            new_graph[:current_num_nodes, :current_num_nodes] = graph_copy # Copy from original copy
            graph_copy = new_graph # Update the working copy

        # Add the new edges to the working copy
        for u, v, value in new_edges_to_add:
            if u < final_num_nodes and v < final_num_nodes:  # Ensure indices are within bounds
                graph_copy[u, v] = value
            else:
                print(f"Warning: Edge ({u}, {v}) indices out of bounds ({final_num_nodes}). Skipping.")

        # --- 4. Final Cleanup of idx2type_copy ---
        # Ensure only successfully added noise rules (and original nodes) remain in the type map
        added_node_indices = set(idx2type_copy.keys()) - set(self.idx2type.keys())
        indices_to_delete = []
        for idx in added_node_indices:
            # Check if it's a rule node that wasn't successfully recorded in noise_rule_info
            if idx2type_copy.get(idx) == 'rule' and idx not in self.noise_rule_info:
                indices_to_delete.append(idx)
            # Optional: Add check for orphaned predicates here if needed later

        if indices_to_delete:
            # print(f"Cleanup: Removing discarded rule indices from noisy_idx2type: {indices_to_delete}")
            for idx in indices_to_delete:
                if idx in idx2type_copy: # Double check before deleting
                    del idx2type_copy[idx]

        # --- 5. Final Cleanup of Isolated Predicates ---
        predicate_indices_in_noisy = {idx for idx, type_ in idx2type_copy.items() if type_ == 'predicate'}
        isolated_predicates_to_delete = []
        final_graph_nodes = graph_copy.shape[0] # Get size of the final graph matrix

        for p_idx in predicate_indices_in_noisy:
             if p_idx < final_graph_nodes: # Ensure index is within graph bounds
                 in_degree = np.sum(graph_copy[:, p_idx]) # Sum of incoming edge weights
                 out_degree = np.sum(graph_copy[p_idx, :]) # Sum of outgoing edge weights
                 if in_degree == 0 and out_degree == 0:
                     isolated_predicates_to_delete.append(p_idx)
             else:
                 # Predicate index is somehow outside the final graph bounds, mark for deletion
                 isolated_predicates_to_delete.append(p_idx)


        if isolated_predicates_to_delete:
             # print(f"Cleanup: Removing isolated predicate indices from noisy_idx2type: {isolated_predicates_to_delete}")
             for idx in isolated_predicates_to_delete:
                if idx in idx2type_copy: # Double check before deleting
                    del idx2type_copy[idx]

        # --- 5. Final Cleanup of Isolated Predicates ---
        # Iterate over a copy of keys as we might modify the dictionary
        current_noisy_predicate_indices = {idx for idx, type_ in idx2type_copy.items() if type_ == 'predicate'}
        isolated_predicates_to_delete = []
        final_graph_nodes = graph_copy.shape[0] # Get size of the final graph matrix

        for p_idx in current_noisy_predicate_indices:
             if p_idx < final_graph_nodes: # Ensure index is within graph bounds
                 # Check if the predicate still exists in the map (might have been deleted as part of a rule cleanup)
                 if p_idx not in idx2type_copy:
                     continue

                 in_degree = np.sum(graph_copy[:, p_idx]) # Sum of incoming edge weights
                 out_degree = np.sum(graph_copy[p_idx, :]) # Sum of outgoing edge weights
                 if in_degree == 0 and out_degree == 0:
                     isolated_predicates_to_delete.append(p_idx)
             else:
                 # Predicate index is somehow outside the final graph bounds, mark for deletion
                 # Check if it still exists before adding
                 if p_idx in idx2type_copy:
                    isolated_predicates_to_delete.append(p_idx)


        if isolated_predicates_to_delete:
             # print(f"Cleanup: Removing isolated predicate indices from noisy_idx2type: {isolated_predicates_to_delete}")
             for idx in isolated_predicates_to_delete:
                 if idx in idx2type_copy: # Double check before deleting
                     del idx2type_copy[idx]

        # --- 6. Index Normalization (Added Step) ---
        # Get final valid node indices after all cleanup
        final_valid_indices = sorted(list(idx2type_copy.keys()))
        num_final_nodes = len(final_valid_indices)

        if num_final_nodes > 0:
            # Create mapping from old (potentially sparse) index to new (dense) index
            old_idx_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(final_valid_indices)}

            # Create new graph and type map with dense indices
            new_noisy_graph = np.zeros((num_final_nodes, num_final_nodes), dtype=int)
            new_noisy_idx2type = {}
            new_noise_rule_info = {} # Also update noise rule info keys

            # Populate new type map and update noise rule info keys
            for old_idx, new_idx in old_idx_to_new_idx.items():
                node_type = idx2type_copy[old_idx]
                new_noisy_idx2type[new_idx] = node_type
                # If this node was a noise rule, update its key in the info dict
                if old_idx in self.noise_rule_info:
                    new_noise_rule_info[new_idx] = self.noise_rule_info[old_idx]

            # Populate new graph using the mapping
            # Iterate through the old graph's non-zero elements for efficiency
            old_rows, old_cols = graph_copy.nonzero()
            for old_u, old_v in zip(old_rows, old_cols):
                # Check if both old indices are still valid (might have been cleaned up)
                if old_u in old_idx_to_new_idx and old_v in old_idx_to_new_idx:
                    new_u = old_idx_to_new_idx[old_u]
                    new_v = old_idx_to_new_idx[old_v]
                    new_noisy_graph[new_u, new_v] = graph_copy[old_u, old_v]

            # Update the instance attributes with the normalized data
            self.noisy_rule_predicate_operation_graph = new_noisy_graph
            self.noisy_idx2type = new_noisy_idx2type
            self.noise_rule_info = new_noise_rule_info # Update noise rule info
        else:
            # Handle edge case where no nodes remain after cleanup
            self.noisy_rule_predicate_operation_graph = np.zeros((0, 0), dtype=int)
            self.noisy_idx2type = {}
            self.noise_rule_info = {} # Clear noise rule info as well

        # Do NOT update the original counts or graph/idx2type here

    def to_ASP(self,
               predicates: list[str] = None,
               m: int = 2,
               largest: int = 1,
               seed: int = None,
               with_noise: bool = False,
               predicate_variables_in: Optional[Dict[int, frozenset[int]]] = None) -> dict: # Added predicate_variables_in
        """
        将当前的 SymTexSample 实例转换为包含规则和事实的字典结构。

        :param predicate_variables_in: 可选的预定义谓词变量字典。如果提供，将优先使用这些变量。

        :param predicates: 可选的谓词名称列表。如果为 None，则自动生成 P0, P1, ...
        :param m: 生成变量时子集的最大大小 (solve 的 m 参数)。
        :param largest: 生成变量时的最大整数值 (solve 的 largest 参数)。
        :param seed: 随机种子，用于变量生成的随机选择。
        :param with_noise: 是否使用添加了噪声的图进行转换 (默认为 False)。
        :param predicate_variables_in: 可选的预定义谓词变量字典。如果提供，将优先使用这些变量。
        :return: 一个字典，包含 "rules" 和 "facts" 列表。
                 - "rules": list[dict], 每个 dict 包含 "head" 和 "body" 列表。
                 - "facts": list[dict], 每个 dict 代表一个事实。
                 每个 predicate dict 包含 "predicateIdx", "strong negation", "default negation" (仅 body), "variables"。
        """
        # --- Select graph and type map based on with_noise ---
        if with_noise:
            if self.noisy_rule_predicate_operation_graph is None or self.noisy_idx2type is None:
                 raise ValueError("噪声图数据尚未生成，无法使用 with_noise=True。请先调用 add_noise()")
            graph = self.noisy_rule_predicate_operation_graph
            current_idx2type = self.noisy_idx2type
        else:
            if self.rule_predicate_operation_graph is None or self.idx2type is None:
                raise ValueError("原始图数据尚未生成，无法转换为 ASP。请先调用 _sample_generate()")
            graph = self.rule_predicate_operation_graph
            current_idx2type = self.idx2type
            # if predicate_variables_in is None: # Only generate if not provided
            #     canonical_predicate_variables: Dict[int, FrozenSet[int]] = {}
            #     predicate_indices_for_canon = sorted([idx for idx, type in current_idx2type.items() if type == 'predicate'])
            #     for p_idx in predicate_indices_for_canon:
            #          # Generate variables for each predicate exactly once
            #          solutions = solve_wrapper(nums=set(), n=1, m=m, largest=largest, must_include_subsets=[])
            #          if solutions:
            #              chosen_solution_tuple = random.choice(solutions)
            #              canonical_predicate_variables[p_idx] = chosen_solution_tuple[0]
            #          else:
            #              print(f"Warning: solve_wrapper failed to generate canonical variable for predicate {p_idx}. Assigning empty set.")
            #              canonical_predicate_variables[p_idx] = frozenset()
            #     predicate_variables_in = canonical_predicate_variables # Use these generated variables


        if seed is not None:
            random.seed(seed) # Set seed for reproducibility in variable selection

        self.predicate_variables.clear() # Clear previous variables if called multiple times

        # Use the selected type map
        predicate_indices = sorted([idx for idx, type in current_idx2type.items() if type == 'predicate'])
        num_predicates_found = len(predicate_indices)

        if predicates is None:
            self.predicates = [f"P{i}" for i in range(num_predicates_found)]
        elif len(predicates) == num_predicates_found:
            self.predicates = predicates
        else:
            raise ValueError(f"提供的谓词列表长度 ({len(predicates)}) 与找到的谓词数量 ({num_predicates_found}) 不匹配。")

        # Note: predicate_map is not strictly needed for the dictionary output,
        # but we keep the predicate generation logic for potential internal use or debugging.
        # predicate_map = {idx: name for idx, name in zip(predicate_indices, self.predicates)}

        # Use the selected type map and graph
        rule_indices = sorted([idx for idx, type in current_idx2type.items() if type == 'rule'])
        result = {"rules": [], "facts": []} # Initialize the result dictionary

        # graph is already selected based on with_noise
        # --- Remove the old internal variable generation and dictionary creation loops ---
        # The entire block from "Variable Generation Loop" to the end of "Generate Facts Dictionary"
        # is replaced by the call to the external function below.

        # Call the external function to generate the ASP dictionary
        result_dict, generated_predicate_variables, generated_predicates_list = generate_asp_dict_from_graph(
            graph=graph,
            idx2type=current_idx2type,
            predicates_in=None, # Pass the input predicates list
            m=m,
            largest=largest,
            seed=seed,
            predicate_variables_in=predicate_variables_in, # Pass the input variables
            # Pass the new flag based on with_noise
            enforce_strict_connectivity=(not with_noise)
        )

        # Update self attributes with the results from the external function
        # Store rules and facts based on the with_noise flag
        if with_noise:
            # --- START: Noise Rule Post-Processing ---
            processed_noise_rules = []
            original_noise_rules = result_dict.get("rules", [])
            original_noise_facts = result_dict.get("facts", []) # Get facts early

            for rule_dict_orig in original_noise_rules:
                # Deep copy to avoid modifying the original result_dict
                rule_dict = {
                    "head": [p.copy() for p in rule_dict_orig.get("head", [])],
                    "body": [p.copy() for p in rule_dict_orig.get("body", [])],
                    "ruleIdx": rule_dict_orig.get("ruleIdx")
                }
                rule_idx = rule_dict.get("ruleIdx")

                # Check if it's a noise rule using noise_rule_info
                if rule_idx is not None and rule_idx in self.noise_rule_info:
                    # It's a noise rule, apply grounding logic
                    head_vars = set()
                    for head_pred in rule_dict.get("head", []):
                        head_vars.update(head_pred.get("variables", []))

                    body_vars = set()
                    for body_pred in rule_dict.get("body", []):
                        body_vars.update(body_pred.get("variables", []))

                    uncovered_head_vars = head_vars - body_vars

                    if uncovered_head_vars and body_vars: # Only proceed if there are uncovered vars AND body vars to replace with
                        body_vars_list = list(body_vars) # Convert to list for random.choice

                        for head_pred_dict in rule_dict.get("head", []):
                            original_vars = head_pred_dict.get("variables", [])
                            original_arity = len(original_vars)
                            new_vars = []
                            modified = False

                            for var in original_vars:
                                if var in uncovered_head_vars:
                                    replacement_var = random.choice(body_vars_list)
                                    new_vars.append(replacement_var)
                                    modified = True
                                else:
                                    new_vars.append(var)

                            if modified:
                                # Ensure arity is maintained (should always be true with this logic)
                                if len(new_vars) != original_arity:
                                     # Revert changes for this specific predicate if arity changed unexpectedly
                                     head_pred_dict["variables"] = original_vars
                                else:
                                     head_pred_dict["variables"] = sorted(new_vars) # Store sorted variables

                    # Add the processed (or original if no grounding needed/possible) rule to the list
                    processed_noise_rules.append(rule_dict)
                else:
                    # Not a noise rule (or ruleIdx missing), add it as is
                    processed_noise_rules.append(rule_dict)

            self.noise_rules = processed_noise_rules
            self.noise_facts = original_noise_facts # Store original facts
            # --- END: Noise Rule Post-Processing ---
        else:
            # For non-noise mode, directly use the results from generate_asp_dict_from_graph
            # which should have used the canonical variables.
            # Post-processing steps (Head Grounding, Body Connectivity) are removed
            # to preserve the canonical variable consistency.
            self.rules = result_dict.get("rules", [])
            self.facts = result_dict.get("facts", [])

        # Update predicate variables and names regardless of noise mode
        # They reflect the state of the last call to to_ASP, potentially including canonical vars
        self.predicate_variables = generated_predicate_variables
        self.predicates = generated_predicates_list

        # Return the original dictionary from generate_asp_dict_from_graph,
        # even if self.noise_rules was corrected.
        return result_dict

    def to_dlv2(self,
                predicates: list[str] = None,
                m: int = 2,
                largest: int = 1,
                seed: int = None,
                use_disjunction: bool = False) -> tuple[list[str], list[str], list[str], dict[str, list[str]], dict[str, list[dict]]]:
        """
        将当前的 SymTexSample 实例转换为 DLV2 格式的字符串列表，并返回原始的字典结构。

        :param predicates: 可选的谓词名称列表。如果为 None，则使用 to_ASP 生成的。
        :param m: 生成变量时子集的最大大小。
        :param largest: 生成变量时的最大整数值。
        :param seed: 随机种子，用于变量生成的随机选择。如果为 None，则使用 self.seed。
        :param use_disjunction: 是否在规则头部使用析取。如果为 False (默认)，
                                具有多个头部的规则将被拆分为多个单头规则。
        :return: 一个元组，包含五个元素：
                 - noiseless_facts_dlv2: list[str], 无噪声事实的 DLV2 字符串列表。
                 - noiseless_rules_dlv2: list[str], 无噪声规则的 DLV2 字符串列表。
                 - noisy_facts_dlv2: list[str], 噪声事实的 DLV2 字符串列表。
                 - noisy_rules_dlv2_by_type: dict[str, list[str]], 按类型分类的噪声规则 DLV2 字符串字典。
                 - original_dicts: dict[str, list[dict]], 包含原始字典 ('noiseless_facts', 'noiseless_rules', 'noisy_facts', 'noisy_rules')。
        """
        current_seed = seed if seed is not None else self.seed
        if current_seed is not None:
            random.seed(current_seed)
            np.random.seed(current_seed)


        # --- 1. Generate Noiseless Data using Canonical Variables ---
        try:
            # Second call to_ASP with noise=False, passing the canonical variables
            # This populates self.rules and self.facts using the fixed variables
            self.to_ASP(predicates=predicates, m=m, largest=largest, seed=current_seed, with_noise=False)
            canonical_predicate_variables = self.predicate_variables.copy() # Store the generated variables

            noiseless_rules_dict = self.rules if self.rules is not None else []
            noiseless_facts_dict = self.facts if self.facts is not None else []
        except ValueError as e:
            print(f"Error: Failed to generate noiseless ASP data using canonical variables: {e}")
            raise e

        # --- 2. Generate Canonical Variables and Noisy Data ---
        # First call to_ASP with noise=True to generate variables based on the full graph
        try:
            # This call generates variables and stores them in self.predicate_variables
            # It also populates self.noise_rules and self.noise_facts
            self.to_ASP(predicates=predicates, m=m, largest=largest, seed=current_seed, with_noise=True, predicate_variables_in=canonical_predicate_variables)
            noisy_rules_dict = self.noise_rules if self.noise_rules is not None else []
            noisy_facts_dict = self.noise_facts if self.noise_facts is not None else [] # Renamed from noisy_rules_fact for clarity
        except ValueError as e:
            print(f"Error: Failed to generate noisy ASP data and canonical variables: {e}")
            raise e


        # Use the final self.predicates list (potentially updated by the noise call)
        final_predicates = self.predicates if self.predicates is not None else []
        # Create a mapping from predicate index to name for easier lookup
        # Ensure the mapping covers all possible indices encountered
        max_idx_noiseless = max([f['predicateIdx'] for f in noiseless_facts_dict] + \
                                [p['predicateIdx'] for r in noiseless_rules_dict for p in r['head']] + \
                                [p['predicateIdx'] for r in noiseless_rules_dict for p in r['body']], default=-1)
        max_idx_noisy = max([p['predicateIdx'] for r in noisy_rules_dict for p in r['head']] + \
                            [p['predicateIdx'] for r in noisy_rules_dict for p in r['body']], default=-1)
        max_idx = max(max_idx_noiseless, max_idx_noisy, len(final_predicates) - 1)

        # Generate default names if final_predicates is shorter than needed
        if len(final_predicates) <= max_idx:
            final_predicates.extend([f"P{i}" for i in range(len(final_predicates), max_idx + 1)])

        idx_to_name = {i: name for i, name in enumerate(final_predicates)}

        # --- Instantiate DLV2 Runner ---
        try:
            dlv2_runner = Dlv2Runner()
        except Dlv2RunnerError as e:
            print(f"Warning: Failed to initialize Dlv2Runner: {e}. Safety checks will be skipped.")
            dlv2_runner = None # Set to None if initialization fails

        # --- 2. Format Noiseless Facts ---
        noiseless_facts_dlv2 = []
        for fact_dict in noiseless_facts_dict:
            # Use the standalone helper function
            formatted_fact = _format_atom_dlv2(fact_dict, idx_to_name, is_fact=True)
            noiseless_facts_dlv2.append(f"{formatted_fact}.")

        # --- 3. Validate and Format Noiseless Rules ---
        noiseless_rules_dlv2 = []
        for original_rule_dict in noiseless_rules_dict:
            rule_idx_noiseless = original_rule_dict.get('ruleIdx', 'N/A')
            context_noiseless = f"noiseless rule (idx {rule_idx_noiseless})"
            # Use the new validation helper function
            validated_rule_dict = _validate_and_modify_rule_for_safety(
                original_rule_dict, dlv2_runner, idx_to_name, use_disjunction, context_noiseless
            )
            # Format the final (potentially modified) rule dict
            noiseless_rules_dlv2.extend(
                _format_rule_dict_to_dlv2(validated_rule_dict, idx_to_name, use_disjunction)
            )

        # --- 4. Format Noisy Facts ---
        noisy_facts_dlv2 = []
        for fact_dict in noisy_facts_dict:
            formatted_fact = _format_atom_dlv2(fact_dict, idx_to_name, is_fact=True)
            noisy_facts_dlv2.append(f"{formatted_fact}.")

        # --- 5. Validate and Format Noisy Rules ---
        noisy_rules_dlv2_by_type: Dict[str, List[str]] = {'a': [], 'b': [], 'c': [], 'd': [], 'e': []}
        noisy_rules_dict_by_type: Dict[str, List[str]] = {'a': [], 'b': [], 'c': [], 'd': [], 'e': []}
        if noisy_rules_dict and hasattr(self, 'noise_rule_info') and self.noise_rule_info:
            for original_noisy_rule_dict in noisy_rules_dict:
                rule_idx_noisy = original_noisy_rule_dict.get("ruleIdx")
                noise_type = self.noise_rule_info.get(rule_idx_noisy) if rule_idx_noisy is not None else None

                if noise_type:
                    # Ensure the noise type exists as a key
                    if noise_type not in noisy_rules_dlv2_by_type:
                        noisy_rules_dlv2_by_type[noise_type] = []

                    context_noisy = f"noisy rule (type {noise_type}, idx {rule_idx_noisy})"
                    # Use the new validation helper function
                    validated_noisy_rule_dict = _validate_and_modify_rule_for_safety(
                        original_noisy_rule_dict, dlv2_runner, idx_to_name, use_disjunction, context_noisy
                    )

                    # Format the final (potentially modified) noisy rule dict
                    formatted_noisy_rules = _format_rule_dict_to_dlv2(
                        validated_noisy_rule_dict, idx_to_name, use_disjunction
                    )
                    noisy_rules_dlv2_by_type[noise_type].extend(formatted_noisy_rules)
                    noisy_rules_dict_by_type[noise_type].append(original_noisy_rule_dict)

        def list_dict_diff(list_a: List[Dict], list_b: List[Dict]) -> List[Dict]:
            a_str_set = set([str(i) for i in list_a])
            b_str_set = set([str(i) for i in list_b])
            diff = a_str_set - b_str_set
            diff_list = [eval(i) for i in diff]
            return diff_list

        # --- 6. Prepare final return values ---
        original_dicts = {
            "noiseless_facts": noiseless_facts_dict,
            "noiseless_rules": noiseless_rules_dict,
            "noisy_facts": list_dict_diff(noisy_facts_dict, noiseless_facts_dict),
            "noisy_rules": noisy_rules_dict_by_type
        }

        return noiseless_facts_dlv2, noiseless_rules_dlv2, noisy_facts_dlv2, noisy_rules_dlv2_by_type, original_dicts
