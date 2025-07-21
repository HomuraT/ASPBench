import random
from typing import List, Set, Dict, Optional, Tuple, FrozenSet # Added Dict, Optional, Tuple, FrozenSet
import numpy as np

from src.utils.graph_structure_utils import topological_sort, check_path_exists # Added check_path_exists
from src.dataset_generation.variable_generation import solve_wrapper # Added import

# Constants for negation values
STRONG_NEGATION_VALUE = 2
DEFAULT_NEGATION_VALUE = 4


def generate_strongly_connected_dag(num_nodes: int, num_edges: int, seed=None) -> np.ndarray:
    """
    生成一个有向无环图 (DAG)，满足：
    1. 节点 0 是唯一汇点(sink)，即没有从 0 出发的边。
    2. 从任意节点 i>0 都存在一条路径到达节点 0。
    3. 总边数最终恰好为 num_edges（若可能）。
    4. 在构造的过程中既可能出现星形，也可能出现多级链或混合分叉。
    :param num_nodes: 节点总数 (>=1)
    :param num_edges: 需要的总边数
    :param seed:      随机种子，可固定结果（可选）
    :return:          adjacency_matrix，大小为 num_nodes x num_nodes 的邻接矩阵
    """
    if seed is not None:
        random.seed(seed)
    # 特殊情况：如果只有 1 个节点，只能有 0 条边
    if num_nodes == 1:
        if num_edges != 0:
            raise ValueError("当 num_nodes == 1 时，图中只能有 0 条边。")
        return np.array([[0]])
    # 计算可行的最小、最大边数
    # 由于 0 不能有出边，其它节点间最多产生 (num_nodes - 1)*(num_nodes - 2)//2 条边
    # 再加上 (num_nodes - 1) 个节点都可以连向 0。
    max_edges = (num_nodes - 1) + (num_nodes - 1)*(num_nodes - 2)//2
    min_edges = (num_nodes - 1)
    if not (min_edges <= num_edges <= max_edges):
        raise ValueError(
            f"给定的边数 num_edges={num_edges} 不可行。\n"
            f"最少需要 {min_edges} 条边才能保证每个节点可达 0,\n"
            f"最多可有 {max_edges} 条边（在节点 0 无出边的 DAG 中）。"
        )
    # 初始化邻接矩阵
    adj_matrix = np.array([[0]*num_nodes for _ in range(num_nodes)])
    # =========== 1) 随机生成"覆盖所有节点到 0"的 in-arborescence ===========
    # 思路：
    #   - 令 0 号节点为根（注意方向是 child->0），
    #   - 依次为 1..(num_nodes-1) 的每个节点随机选一个"父亲" parent ∈ [已接入树节点集合]，
    #     于是就有一条 (node->parent) 边，最终必经若干步到达 0。
    #   - 这样共会产生 (num_nodes - 1) 条边，且因为每个节点最多选择一次父亲，不会产生环。
    #
    # 这样就能同时产生：
    #     - 星形 (所有节点都直接连向 0)
    #     - 链式 (每次都选了上一个新节点做父节点)
    #     - 以及混合分叉
    # 具体哪种，取决于随机过程。
    assigned = {0}  # 已接入到"树"里的节点集合（初始只有 0）
    unassigned = list(range(1, num_nodes))  # 尚未接入树的节点
    random.shuffle(unassigned)  # 随机打乱
    used_edges = 0
    for node in unassigned:
        # 在已经分配了父节点的集合 assigned 中，随机选一个作为它的父亲
        parent = random.choice(list(assigned))
        # 建立 node -> parent
        adj_matrix[node][parent] = 1
        assigned.add(node)
        used_edges += 1
    # 如果已经达到 num_edges 数量就可以返回了（特别是 num_edges == n-1 的情况）
    if used_edges == num_edges:
        return adj_matrix
    # =========== 2) 在已有 DAG 基础上，添加额外边直到到达 num_edges 条 ===========
    # 首先做一个拓扑序，以避免添加造成环
    topo_order = topological_sort(adj_matrix)
    # topo_order 中 0 必然出现在末尾，否则它就有出边。
    # 但我们不强制它出现在末尾，只需保证加边时不致成环即可。
    # 收集所有可能的"无环候选"边 (u->v)：
    #   条件：在 topo_order 中出现的索引 idx(u) < idx(v)，且原先没有这条边
    #   并且 0 不出边，所以去掉 (0->v) 的情况
    idx_in_topo = {node: i for i, node in enumerate(topo_order)}
    candidate_edges = []
    n_ = len(topo_order)
    for i in range(n_):
        u = topo_order[i]
        if u == 0:
            continue  # 0 不能有出边
        for j in range(i + 1, n_):
            v = topo_order[j]
            # 如果还没有 u->v，添加之
            if adj_matrix[u][v] == 0:
                candidate_edges.append((u, v))
    # 打乱候选边的顺序，防止每次都加相同的边
    random.shuffle(candidate_edges)
    need_extra = num_edges - used_edges
    if need_extra > len(candidate_edges):
        # 理论上不会出现，前面检查过 max_edges
        raise ValueError("无法添加足够多的额外边。")
    # 逐一选取 candidate_edges 里前 need_extra 个，并加入图
    for i in range(need_extra):
        u, v = candidate_edges[i]
        adj_matrix[u][v] = 1
    return adj_matrix


def expand_graph(
    M: np.ndarray,
    extra_predicate_num: int = 0,
    extra_edge_num: int = 0,
    max_predicates_per_rule: int = 3, # Add parameter
    seed=None
) -> tuple[np.ndarray, dict[int, str]]:
    """
    扩展有向无环图(仅包含 rule 节点)，包括以下步骤：

      (必需的三步，固定顺序进行)
      1) 对每条原图中 (r_i->r_j) 边，必然添加一个新的 predicate 节点 p_ij，形成 r_i->p_ij->r_j
      2) 对节点 0(r0) 再额外插一个 predicate 节点 p0，形成 r0->p0
      3) 确保每个 rule 节点至少有一条来自新插入 predicate 的 p->r 边

      (可选的额外插入，取决于参数)
      4) 若 extra_predicate_num>0, extra_edge_num>0，则随机插入 extra_predicate_num 个 predicate，
         并再随机插入 extra_edge_num 条 p->r 边(至少保证每个新插 predicate 有一条出边)。

    :param M: 原始的邻接矩阵，表示 N 个 rule 节点 (0 ~ N-1) 之间的有向边关系 (无环, numpy array, shape = (N, N))
    :param extra_predicate_num: 额外插入的 predicate 节点个数 (int)
    :param extra_edge_num: 额外插入的 p->r 形式的边数 (int)
    :param max_predicates_per_rule: 每个规则体部允许的最大谓词数量 (int)
    :param seed: 随机种子，用于复现生成过程 (可选)
    :return: 一个二元组 (final_new_M, idx2type)
        - final_new_M (numpy array): 扩展后的邻接矩阵 (仅保留实际使用到的节点)
        - idx2type (dict): 节点编号 -> 节点类型("rule" 或 "predicate")
    """
    if seed is not None:
        np.random.seed(seed)

    # ===============
    # 第 1 步：初步信息
    # ===============
    if max_predicates_per_rule <= 0:
        raise ValueError("max_predicates_per_rule 必须大于 0")

    N = M.shape[0]  # 原有 rule 节点数
    idx2type = {i: "rule" for i in range(N)}

    # 原图中所有 (r_i->r_j) 边
    edges = [(i, j) for i in range(N) for j in range(N) if M[i, j] == 1]

    # 由于后续会插入新的 predicate，需要在新的邻接矩阵里多预留些空间
    # 先取一个较宽松的上限：原有 N + (每条原图边) + (为 r0 插一个) + (最多为所有 rule 插一个) + extra_predicate_num
    # 看上去很保守，但能满足所有需求
    reserve_size = N + len(edges) + 1 + N + extra_predicate_num
    new_M = np.zeros((reserve_size, reserve_size), dtype=int)

    # 用于记录下一个可用 predicate 节点编号
    next_new_pred_index = N

    # 用于记录每个 rule 节点的入度 (来自 predicate)
    rule_in_degree = {r_idx: 0 for r_idx in range(N)}

    # ===============
    # 第 2 步：对每条原图中 (r_i->r_j) 边，插入 p_ij => r_i->p_ij->r_j
    # ==============
    for _ in range(len(edges)):
        idx2type[next_new_pred_index] = "predicate"
        next_new_pred_index += 1

    # ===============
    # 第 3 步：给节点 0(r0) 再额外插一个 predicate 节点 p0, r0->p0
    # ==============
    idx2type[next_new_pred_index] = "predicate"
    next_new_pred_index += 1

    # ===============
    # 第 4 步：正式把原图 (r_i->r_j) 边扩展成 r_i->p->r_j
    # ==============
    # 回想一下：原图边对应的 predicate，按顺序从 N 开始，连续 len(edges) 个
    pred_for_edges_start = N
    for (i, j) in edges: # i = source rule, j = target rule
        p = pred_for_edges_start
        new_M[i, p] = 1  # r_i -> p

        # Check constraint before adding p -> r_j
        if rule_in_degree[j] >= max_predicates_per_rule:
            raise ValueError(
                f"无法添加必要的边 p{p}->r{j}，因为规则 r{j} 的入度 "
                f"({rule_in_degree[j]}) 将超过 max_predicates_per_rule "
                f"({max_predicates_per_rule})。请尝试更大的 max_predicates_per_rule 或更稀疏的初始规则图。"
            )
        new_M[p, j] = 1  # p -> r_j
        rule_in_degree[j] += 1 # Increment in-degree for rule j
        pred_for_edges_start += 1

    # ===============
    # 第 5 步：对 r0 的那个 predicate p0
    # ==============
    p0 = pred_for_edges_start
    new_M[0, p0] = 1  # r0 -> p0
    pred_for_edges_start += 1

    # ===============
    # 第 6 步：确保每个 rule 节点都有至少一条来自"新插入 predicate"的 p->r
    #         若没有，则补一个 predicate。
    # ==============
    for r_idx in range(N):
        # 检查是否有 p (>=N) 指向 r_idx
        # 检查是否有 p (>=N) 指向 r_idx (注意：此时 pred_for_edges_start 包含了为 r0 添加的 p0)
        # 我们需要检查所有已创建的 predicate (N 到 next_new_pred_index-1)
        # 但实际上，我们只关心由步骤4和步骤6创建的谓词是否指向r_idx
        # 一个更准确的方法是检查 rule_in_degree[r_idx] 是否仍然为 0
        # has_new_pred_inbound = np.any(new_M[np.arange(N, next_new_pred_index), r_idx] == 1) # Check all current predicates
        if rule_in_degree[r_idx] == 0: # If no predicate has pointed to this rule yet
            # Check constraint before adding the ensuring edge
            if rule_in_degree[r_idx] >= max_predicates_per_rule: # This check is technically redundant if rule_in_degree is 0, but good for safety
                 raise ValueError(
                     f"无法添加确保连通性的边 p_new->r{r_idx}，因为规则 r{r_idx} 的入度 "
                     f"({rule_in_degree[r_idx]}) 将超过 max_predicates_per_rule "
                     f"({max_predicates_per_rule})。这通常发生在 max_predicates_per_rule=0 或非常小的情况下。"
                 )
            p_new = next_new_pred_index
            idx2type[p_new] = "predicate"
            new_M[p_new, r_idx] = 1
            rule_in_degree[r_idx] += 1 # Increment in-degree
            next_new_pred_index += 1

    # ===============
    # 第 7 步：若要插入额外的 predicate 和 p->r 边，则执行
    # ==============
    if extra_predicate_num > 0 or extra_edge_num > 0:
        if extra_predicate_num > 0 and extra_predicate_num > extra_edge_num:  # 仅当需要插入 predicate 时，才强制 extra_edge_num >= extra_predicate_num
            raise ValueError("extra_predicate_num 必须小于等于 extra_edge_num，才能保证每个新插入的 predicate 至少连一条边。")

        # 先新增 extra_predicate_num 个谓词
        extra_predicate_indices = []
        for _ in range(extra_predicate_num):
            idx2type[next_new_pred_index] = "predicate"
            extra_predicate_indices.append(next_new_pred_index)
            next_new_pred_index += 1

        # 再插入 extra_edge_num 条 p->r，同时考虑 max_predicates_per_rule 限制
        used_extra_edges = 0
        if extra_edge_num > 0:
            # --- 7.1 保证每个新 predicate 至少有1条 p->r (如果 extra_predicate_num > 0) ---
            if extra_predicate_indices:
                shuffled_rule_indices = list(range(N))
                np.random.shuffle(shuffled_rule_indices)

                for p_idx in extra_predicate_indices:
                    edge_added_for_p = False
                    for r_idx in shuffled_rule_indices:
                        # 检查规则入度是否已满，边是否已存在，以及是否会形成环
                        would_create_cycle = check_path_exists(new_M[:next_new_pred_index, :next_new_pred_index], r_idx, p_idx)
                        if rule_in_degree[r_idx] < max_predicates_per_rule and new_M[p_idx, r_idx] == 0 and not would_create_cycle:
                            new_M[p_idx, r_idx] = 1
                            rule_in_degree[r_idx] += 1
                            used_extra_edges += 1
                            edge_added_for_p = True
                            break # 已为该 predicate 添加边，处理下一个

                    if not edge_added_for_p:
                         # 尝试了所有规则，都无法为这个新 predicate 添加出边
                         raise ValueError(
                             f"无法为新谓词 p{p_idx} 添加至少一条出边。这可能是因为所有规则的入度都已达到 "
                             f"max_predicates_per_rule ({max_predicates_per_rule}) 的限制，或者添加任何可能的边都会导致环路。"
                             f"请尝试增加 max_predicates_per_rule 或减少 extra_predicate_num/extra_edge_num。"
                         )

            # --- 7.2 若还有剩余边，则可继续随机插 ---
            remain_edges = extra_edge_num - used_extra_edges
            added_count = 0
            attempts = 0
            # 设定一个合理的尝试上限，例如 5 倍于所需边数乘以规则数
            max_attempts = remain_edges * N * 5 + 100 # Added a base attempts number

            # 确定用于剩余边的谓词选择池 (所有已创建的谓词)
            all_created_predicate_indices = [idx for idx, type_ in idx2type.items() if type_ == "predicate" and idx < next_new_pred_index]

            if remain_edges > 0 and not all_created_predicate_indices:
                 print(f"Warning: Trying to add {remain_edges} extra edges, but no predicates exist yet.") # Should not happen if N>0
                 remain_edges = 0 # Cannot add edges

            if remain_edges > 0:
                predicate_pool_for_remaining = np.array(all_created_predicate_indices)

                while added_count < remain_edges and attempts < max_attempts:
                    attempts += 1
                    p_idx = np.random.choice(predicate_pool_for_remaining)

                    # 找出所有入度未满的规则
                    available_rules = [r for r in range(N) if rule_in_degree[r] < max_predicates_per_rule]

                    if not available_rules:
                        # 所有规则的入度都满了，无法再添加边
                        print(f"Warning: All rules reached max_predicates_per_rule ({max_predicates_per_rule}). "
                              f"Could not add remaining {remain_edges - added_count} extra edges.")
                        break # 退出循环

                    r_idx = np.random.choice(available_rules)

                    # 只有当边不存在，且不会形成环时才添加，并计数
                    would_create_cycle = check_path_exists(new_M[:next_new_pred_index, :next_new_pred_index], r_idx, p_idx)
                    if new_M[p_idx, r_idx] == 0 and not would_create_cycle:
                        new_M[p_idx, r_idx] = 1
                        rule_in_degree[r_idx] += 1
                        added_count += 1
                        used_extra_edges += 1 # Also update the total count

                # 检查是否成功添加了所有需要的边
                if added_count < remain_edges:
                      # 如果未能添加所有请求的剩余边，则抛出错误
                      raise ValueError(
                          f"未能添加所有请求的 {extra_edge_num} 条额外边（只添加了 {used_extra_edges} 条中的 {added_count} 条剩余边）。"
                          f"这很可能是由于 max_predicates_per_rule ({max_predicates_per_rule}) 的限制，或者因为添加剩余的边会引入环路，"
                          f"在尝试 {attempts} 次后仍无法找到足够的可添加边。"
                          f"请尝试增加 max_predicates_per_rule 或减少 extra_edge_num。"
                      )
                      # print(f"Warning: Could only add {added_count} / {remain_edges} requested remaining extra edges "
                      #       f"after {attempts} attempts, likely due to max_predicates_per_rule limit or cycle prevention. " # Added cycle prevention note
                      #       f"Total extra edges added: {used_extra_edges}.") # Replaced print with raise

    # ===============
    # 第 8 步：裁剪出实际使用到的节点范围
    # ==============
    final_size = next_new_pred_index
    final_new_M = new_M[:final_size, :final_size].copy()

    # 返回
    return final_new_M, idx2type

def apply_negations(
    adj_matrix: np.ndarray,
    idx2type: dict[int, str],
    strong_negation_prob: float,
    default_negation_prob: float,
    seed=None
) -> np.ndarray:
    """
    根据给定的概率，为 predicate 节点添加 strong negation 和 default negation 标记，
    并将这些标记应用到与这些 predicate 相关的边上。

    Negation 标记使用位运算 OR 操作添加到边的值上：
    - Strong negation:  |= 2
    - Default negation: |= 4

    :param adj_matrix: 原始的邻接矩阵 (numpy array)
    :param idx2type: 节点编号 -> 节点类型 ("rule" 或 "predicate") 的字典
    :param strong_negation_prob: 添加 strong negation 的概率 (0.0 到 1.0)
    :param default_negation_prob: 添加 default negation 的概率 (0.0 到 1.0)
    :param seed: 随机种子，用于复现 (可选)
    :return: 一个新的邻接矩阵，其中与 predicate 相关的边可能已根据 negation 规则被修改
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed) # Also seed numpy if using numpy's random functions

    num_nodes = adj_matrix.shape[0]
    new_adj_matrix = adj_matrix.copy() # 创建副本，不修改原矩阵

    predicate_negation_values = {} # 存储每个 predicate 的 negation 值

    # 1. 确定每个 predicate 的 negation 值
    for node_idx, node_type in idx2type.items():
        if node_idx >= num_nodes: # Skip indices out of bounds for adj_matrix
            continue
        if node_type == "predicate":
            negation_value = 0
            # Ensure probabilities are within valid range [0, 1]
            current_strong_prob = max(0.0, min(1.0, strong_negation_prob))
            current_default_prob = max(0.0, min(1.0, default_negation_prob))

            if random.random() < current_strong_prob:
                negation_value |= STRONG_NEGATION_VALUE
            if random.random() < current_default_prob:
                negation_value |= DEFAULT_NEGATION_VALUE

            if negation_value > 0:
                 predicate_negation_values[node_idx] = negation_value

    # 2. 将 negation 值应用到相关边上
    for p_idx, neg_val in predicate_negation_values.items():
        if p_idx >= num_nodes: # Skip indices out of bounds for adj_matrix
            continue
        # 遍历所有可能的节点 j
        for j in range(num_nodes):
            # 检查入边 (j, p_idx) - 确保 j 也在范围内
            if j < num_nodes and adj_matrix[j, p_idx] != 0: # 检查原图中是否存在边
                new_adj_matrix[j, p_idx] |= neg_val

            # 检查出边 (p_idx, j) - 确保 j 也在范围内
            if j < num_nodes and adj_matrix[p_idx, j] != 0: # 检查原图中是否存在边
                new_adj_matrix[p_idx, j] |= neg_val

    return new_adj_matrix


def get_sorted_variable_list(var_set: FrozenSet[int]) -> list[int]: # Changed type hint
    """Helper to convert a frozenset of integers to a sorted list."""
    if not var_set:
        return []
    return sorted(list(var_set))

def get_modified_noise_edge_flag(initial_flag: int) -> int:
    """
    根据谓词的初始否定状态，随机生成一个不同的否定状态用于噪声边。

    转换规则:
    - 原始是 Strong (3) 或 Default (5) -> 随机变成 Strong+Default (7) 或 None (1)
    - 原始是 None (1) 或 Strong+Default (7) -> 随机变成 Strong (3) 或 Default (5)

    :param initial_flag: 谓词的初始否定标志 (应包含基础值 1)
    :return: 一个修改后的否定标志
    """
    # 确保基础值是 1
    initial_flag |= 1

    if initial_flag == (1 | STRONG_NEGATION_VALUE) or initial_flag == (1 | DEFAULT_NEGATION_VALUE):
        # Case 1: Original was Strong only or Default only
        possible_new_flags = [1 | STRONG_NEGATION_VALUE | DEFAULT_NEGATION_VALUE, 1] # Strong+Default or None
        return random.choice(possible_new_flags)
    elif initial_flag == 1 or initial_flag == (1 | STRONG_NEGATION_VALUE | DEFAULT_NEGATION_VALUE):
        # Case 2: Original was None or Strong+Default
        possible_new_flags = [1 | STRONG_NEGATION_VALUE, 1 | DEFAULT_NEGATION_VALUE] # Strong only or Default only
        return random.choice(possible_new_flags)
    else:
        # Should not happen with current flags, but return initial as fallback
        print(f"Warning: Unexpected initial_flag {initial_flag} in get_modified_noise_edge_flag. Returning initial.")
        return initial_flag

# --- New Function: generate_asp_dict_from_graph ---
def generate_asp_dict_from_graph(
    graph: np.ndarray,
    idx2type: Dict[int, str],
    predicates_in: Optional[List[str]] = None, # Renamed from predicates
    m: int = 2,
    largest: int = 1,
    seed: Optional[int] = None,
    predicate_variables_in: Optional[Dict[int, FrozenSet[int]]] = None, # New parameter
    enforce_strict_connectivity: bool = False # New parameter for non-noise check
) -> Tuple[Dict[str, List[Dict]], Dict[int, FrozenSet[int]], List[str]]:
    """
    将给定的图结构转换为包含规则和事实的字典结构 (ASP-like)。

    :param enforce_strict_connectivity: 如果为 True，则在生成规则时执行严格的 Body 连通性检查，
                                        并在发现不连通时跳过该规则（主要用于无噪声模式）。

    :param predicate_variables_in: 可选的预定义谓词变量字典。如果提供，将使用这些变量，并且不会为这些谓词重新生成变量。

    :param graph: 邻接矩阵 (numpy array)，包含规则和谓词节点，边值可能包含否定标记。
    :param idx2type: 节点编号 -> 节点类型 ("rule" 或 "predicate") 的字典。
    :param predicates_in: 可选的谓词名称列表。如果为 None，则自动生成 P0, P1, ...
    :param m: 生成变量时子集的最大大小 (solve 的 m 参数)。
    :param largest: 生成变量时的最大整数值 (solve 的 largest 参数)。
    :param seed: 随机种子，用于变量生成的随机选择。
    :return: 一个元组，包含:
             - result_dict: {"rules": list[dict], "facts": list[dict]}
             - predicate_variables: Dict[int, FrozenSet[int]] 谓词索引到其变量集合的映射。
             - generated_predicates_list: List[str] 生成或使用的谓词名称列表。
    """
    if seed is not None:
        random.seed(seed) # Set seed for reproducibility in variable selection

    # Initialize local variable storage, potentially from input
    predicate_variables: Dict[int, FrozenSet[int]] = {}
    if predicate_variables_in is not None:
        predicate_variables.update(predicate_variables_in) # Copy pre-defined variables

    predicate_indices = sorted([idx for idx, type in idx2type.items() if type == 'predicate'])
    # Create mapping from graph node index to 0-based index
    node_idx_to_0based_idx = {node_idx: i for i, node_idx in enumerate(predicate_indices)}
    num_predicates_found = len(predicate_indices)

    if predicates_in is None:
        generated_predicates_list = [f"P{i}" for i in range(num_predicates_found)]
    elif len(predicates_in) == num_predicates_found:
        generated_predicates_list = predicates_in
    else:
        raise ValueError(f"提供的谓词列表长度 ({len(predicates_in)}) 与找到的谓词数量 ({num_predicates_found}) 不匹配。")

    rule_indices = sorted([idx for idx, type in idx2type.items() if type == 'rule'])
    result = {"rules": [], "facts": []} # Initialize the result dictionary

    num_total_nodes = graph.shape[0]

    # --- Variable Generation Loop ---
    for rule_idx in rule_indices:
        if rule_idx >= num_total_nodes: continue

        current_rule_conclusion_pred_indices = []
        current_rule_body_pred_indices = []

        # Identify conclusion predicates for this rule
        for p_idx in predicate_indices:
            if p_idx >= num_total_nodes: continue
            if graph[rule_idx, p_idx] > 0:
                current_rule_conclusion_pred_indices.append(p_idx)

        # Identify body predicates for this rule
        for p_idx in predicate_indices:
            if p_idx >= num_total_nodes: continue
            if graph[p_idx, rule_idx] > 0:
                current_rule_body_pred_indices.append(p_idx)

        # --- Generate variables for conclusion predicates ---
        conclusion_vars_union: Set[int] = set() # Use Set[int] for union operations
        for p_idx in current_rule_conclusion_pred_indices:
            # Check if variables are already defined for this predicate
            if p_idx in predicate_variables:
                conclusion_vars_union.update(predicate_variables[p_idx])
                continue # Skip generation if already defined

            # Variables not defined, proceed with generation
            existing_vars_fs = predicate_variables.get(p_idx) # Should be None here, but check for safety
            must_include = [existing_vars_fs] if existing_vars_fs else []
            solutions = solve_wrapper(nums=set(), n=1, m=m, largest=largest, must_include_subsets=must_include)
            if solutions:
                chosen_solution_tuple = random.choice(solutions)
                new_vars_fs = chosen_solution_tuple[0]
                # Correct indentation for lines inside 'if solutions:'
                predicate_variables[p_idx] = new_vars_fs
                conclusion_vars_union.update(new_vars_fs) # Update the set
            else: # Correct indentation for 'else:' block
                # solve_wrapper failed to find a solution.
                # This could be because existing_vars_fs was incompatible or simply no solution exists.
                # If existing_vars_fs exists, it might be invalid or just unlucky.
                # To be safe, let's reset the variables for this predicate if solve fails.
                # We also don't update conclusion_vars_union with potentially invalid existing_vars.
                # Correct indentation for lines inside 'else:'
                if existing_vars_fs:
                    print(f"Warning: solve_wrapper failed for conclusion predicate {p_idx} with existing vars {existing_vars_fs}. Resetting variables for this predicate.")
                    predicate_variables[p_idx] = frozenset()
                # If existing_vars_fs didn't exist and solve failed, predicate_variables[p_idx] remains unset (or empty).
                # No update to conclusion_vars_union in case of failure.


        # --- Generate variables for body predicates ---
        if current_rule_body_pred_indices:
            # Sort body predicate indices to ensure consistent assignment order
            sorted_body_pred_indices = sorted(current_rule_body_pred_indices)

            must_include_body: List[FrozenSet[int]] = [] # List of frozensets
            # Collect must_include based on the *sorted* order, only adding non-empty sets
            # Also, identify predicates whose variables need to be generated
            predicates_to_generate_vars_for = []
            for p_idx in sorted_body_pred_indices:
                existing_vars_fs = predicate_variables.get(p_idx)
                if existing_vars_fs:
                    must_include_body.append(existing_vars_fs)
                else:
                    # Mark this predicate for variable generation if not already defined
                    predicates_to_generate_vars_for.append(p_idx)

            num_conditions_to_generate = len(predicates_to_generate_vars_for)

            # Only call solve_wrapper if there are body predicates needing variables
            if num_conditions_to_generate > 0:
                # Pass conclusion_vars_union (Set[int]) to solve_wrapper
                # Note: must_include_body now only contains non-empty sets that must be included.
                solutions_body = solve_wrapper(nums=conclusion_vars_union, n=len(sorted_body_pred_indices), m=m, largest=largest, must_include_subsets=must_include_body)

                if solutions_body:
                    chosen_solution_tuple_body = random.choice(solutions_body) # Tuple of frozensets

                    # Ensure the number of solutions matches the number of predicates *needing* variables
                    if len(chosen_solution_tuple_body) == num_conditions_to_generate:
                        # Assign each generated variable set to its corresponding body predicate that needed it
                        for i, p_idx in enumerate(predicates_to_generate_vars_for):
                            assigned_vars = chosen_solution_tuple_body[i]
                            predicate_variables[p_idx] = assigned_vars
                    else:
                        # Handle potential mismatch
                        pass
                    # Fallback: Assign the first set to all, or handle error differently
                    if chosen_solution_tuple_body:
                         fallback_vars = chosen_solution_tuple_body[0]
                         for p_idx in sorted_body_pred_indices:
                             # Check if variable already exists before overwriting with fallback
                             if p_idx not in predicate_variables:
                                 predicate_variables[p_idx] = fallback_vars
                    # Consider raising an error instead of fallback if strictness is required

            # else: If solutions_body is empty, variables might remain unassigned or keep previous values.
            # This part of the logic remains the same.

    # --- ASP Dictionary Generation Loop ---
    # Import defaultdict here if not already imported at the top level
    from collections import defaultdict

    for rule_idx in rule_indices:
        if rule_idx >= num_total_nodes: continue

        # --- START: Body Connectivity Check (if enforce_strict_connectivity is True) ---
        if enforce_strict_connectivity:
            current_body_pred_indices_for_check = []
            # Find body predicates for the current rule
            for p_idx_check in predicate_indices:
                # Check bounds and if edge exists from predicate to rule
                if p_idx_check < num_total_nodes and graph[p_idx_check, rule_idx] > 0:
                    current_body_pred_indices_for_check.append(p_idx_check)

            # Only perform check if there are multiple body predicates
            if len(current_body_pred_indices_for_check) > 1:
                all_body_vars_check = set()
                var_graph_check = defaultdict(set)

                # Build variable graph for the body predicates
                for p_idx_check in current_body_pred_indices_for_check:
                    variables_set = predicate_variables.get(p_idx_check, frozenset()) # p_idx_check is correct here as it's defined locally
                    variables_list = list(variables_set)
                    all_body_vars_check.update(variables_list)
                    # Add edges between variables within the same predicate
                    for i in range(len(variables_list)):
                        for j in range(i + 1, len(variables_list)):
                            u, v = variables_list[i], variables_list[j]
                            var_graph_check[u].add(v)
                            var_graph_check[v].add(u)

                # Find connected components using BFS if variables exist
                if all_body_vars_check:
                    visited_check = set()
                    components_check = []
                    for var_check in all_body_vars_check:
                        if var_check not in visited_check:
                            component = set()
                            queue = [var_check]
                            visited_check.add(var_check)
                            while queue:
                                curr = queue.pop(0)
                                component.add(curr)
                                # Explore neighbors in the variable graph
                                for neighbor in var_graph_check.get(curr, set()):
                                    if neighbor not in visited_check:
                                        visited_check.add(neighbor)
                                        queue.append(neighbor)
                            components_check.append(component)

                    # If more than one component, the body is disconnected
                    if len(components_check) > 1:
                        # print(f"Error: Rule {rule_idx} (strict connectivity enforced) has a disconnected body. Variables: {all_body_vars_check}, Components: {components_check}. Skipping rule.")
                        continue # Skip processing this rule further
        # --- END: Body Connectivity Check ---

        rule_dict = {"head": [], "body": []}

        # --- Process Head ---
        head_predicates_found = False
        for p_idx in predicate_indices:
             if p_idx >= num_total_nodes: continue
             edge_value = graph[rule_idx, p_idx]
             if edge_value > 0:
                 head_predicates_found = True
                 pred_vars_set = predicate_variables.get(p_idx, frozenset())
                 variables = get_sorted_variable_list(pred_vars_set)

                 has_strong = (edge_value & STRONG_NEGATION_VALUE) > 0
                 has_default = (edge_value & DEFAULT_NEGATION_VALUE) > 0
                 # Head is negated (-P) ONLY if EITHER strong OR default negation is present, BUT NOT BOTH.
                 is_negated = (has_strong and not has_default) or (not has_strong and has_default)

                 mapped_idx = node_idx_to_0based_idx[p_idx]
                 rule_dict["head"].append({
                     "predicateIdx": mapped_idx, # Use 0-based index
                     "strong negation": bool(is_negated),
                     "variables": variables
                 })

        if not head_predicates_found: continue # Skip rule if no head predicate

        # --- Process Body ---
        body_predicate_indices = [] # Collect body predicate indices first
        for p_idx in predicate_indices:
            if p_idx >= num_total_nodes: continue
            if graph[p_idx, rule_idx] > 0:
                body_predicate_indices.append(p_idx)

        num_conditions = len(body_predicate_indices)
        all_have_default_negation = False
        if num_conditions > 0:
            all_have_default_negation = all(
                (graph[p_idx, rule_idx] & DEFAULT_NEGATION_VALUE) > 0
                for p_idx in body_predicate_indices
            )

        # Sort predicate indices to ensure consistent "first" predicate for the special rule
        body_preds_info = []
        for p_idx in body_predicate_indices:
            edge_value = graph[p_idx, rule_idx]
            pred_vars_set = predicate_variables.get(p_idx, frozenset())
            variables = get_sorted_variable_list(pred_vars_set)
            body_preds_info.append({
                "p_idx": p_idx,
                "edge_value": edge_value,
                "variables": variables
            })

        # Sort based on predicate index for deterministic order
        body_preds_info.sort(key=lambda x: x["p_idx"])

        for i, pred_info in enumerate(body_preds_info):
            p_idx = pred_info["p_idx"]
            edge_value = pred_info["edge_value"]
            variables = pred_info["variables"]

            apply_special_rule = all_have_default_negation and i == 0

            if apply_special_rule:
                # Apply special rules ONLY to the first predicate if all have default negation
                if edge_value == (1 | DEFAULT_NEGATION_VALUE): # Default negation only (5) -> becomes strong negation
                    body_strong_negation = True
                    body_default_negation = False # Original default is overridden
                elif edge_value == (1 | DEFAULT_NEGATION_VALUE | STRONG_NEGATION_VALUE): # Default and strong negation (7) -> becomes no negation
                    body_strong_negation = False
                    body_default_negation = False # Original negations are cancelled
                else:
                    # Fallback: Apply standard logic (should not happen if all_have_default_negation is true)
                    body_strong_negation = (edge_value & STRONG_NEGATION_VALUE) > 0
                    body_default_negation = (edge_value & DEFAULT_NEGATION_VALUE) > 0
            else:
                # Apply standard rules for negation flags
                body_strong_negation = (edge_value & STRONG_NEGATION_VALUE) > 0
                body_default_negation = (edge_value & DEFAULT_NEGATION_VALUE) > 0

            mapped_idx = node_idx_to_0based_idx[p_idx]
            rule_dict["body"].append({
                "predicateIdx": mapped_idx, # Use 0-based index
                "strong negation": bool(body_strong_negation),
                "default negation": bool(body_default_negation),
                "variables": variables
            })

        # --- Check and enforce body variable coverage over head variables ---
        head_vars = set()
        for head_pred in rule_dict.get("head", []):
            head_vars.update(head_pred.get("variables", []))

        body_vars = set()
        for body_pred in rule_dict.get("body", []):
            body_vars.update(body_pred.get("variables", []))

        uncovered_vars = head_vars - body_vars

        if uncovered_vars and body_vars: # Only replace if there are uncovered vars and body vars to replace with
            body_vars_list = list(body_vars) # Convert to list for random.choice
            for head_pred in rule_dict.get("head", []):
                head_pred_vars = head_pred.get("variables", [])
                for i in range(len(head_pred_vars)):
                    if head_pred_vars[i] in uncovered_vars:
                        # Replace with a random variable from the body
                        head_pred_vars[i] = random.choice(body_vars_list)
        # --- End of coverage check ---

        # Add the completed rule dictionary to the results
        rule_dict["ruleIdx"] = rule_idx # Add the rule index

        result["rules"].append(rule_dict)


    # --- Generate Facts Dictionary ---
    for p_idx in predicate_indices:
        if p_idx >= num_total_nodes: continue # Ensure index is valid

        # Check if this predicate has any incoming edge from *any* rule
        has_incoming_rule_edge = False
        for r_idx in rule_indices:
            if r_idx < num_total_nodes and graph[r_idx, p_idx] > 0:
                has_incoming_rule_edge = True
                break

        # If no incoming rule edge, generate a fact dictionary
        if not has_incoming_rule_edge:
            pred_vars_set = predicate_variables.get(p_idx, frozenset())
            variables = get_sorted_variable_list(pred_vars_set)

            # Determine fact negation based on the first outgoing edge to a rule
            fact_strong_negation = False # Default to positive fact

            outgoing_rule_targets = []
            for target_r_idx in rule_indices:
                 if target_r_idx < num_total_nodes and graph[p_idx, target_r_idx] > 0:
                     outgoing_rule_targets.append(target_r_idx)

            if outgoing_rule_targets:
                min_target_r_idx = min(outgoing_rule_targets)
                edge_value = graph[p_idx, min_target_r_idx]

                has_strong_out = (edge_value & STRONG_NEGATION_VALUE) > 0
                has_default_out = (edge_value & DEFAULT_NEGATION_VALUE) > 0

                # Apply the user-specified rules for fact generation based on negation of the *outgoing* edge
                if has_default_out:
                    if has_strong_out: # Default and Strong Negation (not -P in rule body) -> positive fact
                        fact_strong_negation = False
                    else: # Default Negation Only (not P in rule body) -> strongly negated fact
                        fact_strong_negation = True
                elif has_strong_out: # Strong Negation Only (-P in rule body) -> strongly negated fact
                    fact_strong_negation = True
                # else: No negation (P in rule body) -> positive fact (negation remains False)
            # else: If the predicate has no outgoing edges to rules either, generate a positive fact.

            mapped_idx = node_idx_to_0based_idx[p_idx]
            result["facts"].append({
                "predicateIdx": mapped_idx, # Use 0-based index
                "strong negation": bool(fact_strong_negation),
                "variables": variables
            })

    return result, predicate_variables, generated_predicates_list
