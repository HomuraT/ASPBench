import functools
from itertools import combinations
from typing import List, Set, FrozenSet, Tuple, Optional, Iterable


def generate_subsets_extended(largest: int, m: int) -> List[FrozenSet[int]]:
    """
    Generates all subsets of integers in the range [0, largest] with sizes between 1 and m (inclusive).

    :param largest: The maximum integer allowed in the subsets.
    :type largest: int
    :param m: The maximum size of the subsets to generate.
    :type m: int
    :return: A list of frozensets, each representing a valid subset.
    :rtype: List[FrozenSet[int]]
    """
    candidates = []
    big_range = list(range(largest + 1))
    for size in range(1, m + 1):
        for combo in combinations(big_range, size):
            candidates.append(frozenset(combo))
    return candidates


def is_connected(subsets: List[FrozenSet[int]]) -> bool:
    """
    Checks if a list of subsets is connected based on shared elements.
    Two subsets are considered connected if their intersection is non-empty.
    Uses DFS to check if the graph formed by subsets (nodes) and connections (edges)
    has only one connected component.

    :param subsets: A list of frozensets, where each frozenset represents a node.
    :type subsets: List[FrozenSet[int]]
    :return: True if the subsets form a single connected component, False otherwise.
    :rtype: bool
    """
    n = len(subsets)
    if n <= 1:
        # 0组或1组时，视为连通
        return True

    # 构造邻接表
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if subsets[i].intersection(subsets[j]):
                adj[i].append(j)
                adj[j].append(i)

    # 用 DFS 检查连通分量是否只有一个
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 1

    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if not visited[nxt]:
                visited[nxt] = True
                stack.append(nxt)
                count += 1

    return (count == n)


@functools.lru_cache(maxsize=None)
def solve(nums_frozen: FrozenSet[int],
          n: int,
          m: int,
          largest: int,
          must_include_subsets_frozen: Optional[Tuple[FrozenSet[int], ...]] = None) -> List[Tuple[FrozenSet[int], ...]]:
    """
    Finds combinations of n subsets (chosen from the range [0..largest], with sizes [1..m])
    that satisfy connectivity and coverage constraints. Uses backtracking and caching.

    Constraints:
    1. If nums_frozen is not empty, the union of the chosen n subsets must cover all elements in nums_frozen.
    2. The chosen n subsets must be connected (see `is_connected`).
    3. If must_include_subsets_frozen is provided, these subsets must be part of the chosen n subsets.

    :param nums_frozen: A frozenset of integers that must be covered by the union of the chosen subsets.
                        If empty, the coverage check is skipped.
    :type nums_frozen: FrozenSet[int]
    :param n: The exact number of subsets to choose.
    :type n: int
    :param m: The maximum size allowed for each chosen subset.
    :type m: int
    :param largest: The maximum integer value allowed within the subsets.
    :type largest: int
    :param must_include_subsets_frozen: An optional tuple of frozensets that *must* be included
                                        in the final solution. Assumed to be pre-validated and sorted.
    :type must_include_subsets_frozen: Optional[Tuple[FrozenSet[int], ...]]
    :raises ValueError: If the number of `must_include_subsets` exceeds `n`, or if any fixed subset
                        has an invalid size or contains elements outside the allowed range.
    :return: A list of solutions. Each solution is a tuple containing n frozensets.
    :rtype: List[Tuple[FrozenSet[int], ...]]

    Note:
        - This function is cached using lru_cache. Ensure hashable arguments are passed (like frozensets and tuples).
        - The function assumes `must_include_subsets_frozen` (if provided) has already been validated
          for size, element range, and sorted tuple structure by the caller (e.g., `solve_wrapper`).
          Internal checks are kept as a safeguard.
    """
    # Convert frozenset back to set for easier manipulation (e.g., issubset check)
    nums: Set[int] = set(nums_frozen) if nums_frozen else set()

    # Convert the tuple of frozensets back to a list for internal backtracking logic
    must_include_subsets: List[FrozenSet[int]] = list(must_include_subsets_frozen) if must_include_subsets_frozen else []

    # 检查固定子集的合法性 (现在在调用者处完成，或保持在这里作为双重检查)
    # 这里假设传入的 must_include_subsets_frozen 已经是合法的 frozenset 元组
    if len(must_include_subsets) > n:
        raise ValueError(f"必须包含的子集数量超过了目标 {n}。但 {len(must_include_subsets)} > {n}")
    for fs in must_include_subsets:
        # Restore original check: size must be between 1 and m (inclusive)
        if len(fs) == 0 or len(fs) > m:
            raise ValueError(f"固定子集 {fs} 大小超出允许范围 [1..{m}]。")
        if any((x < 0 or x > largest) for x in fs):
            raise ValueError(f"固定子集 {fs} 中有数字不在 [0..{largest}] 范围内。")

    # Generate candidate subsets to choose from to complete the solution
    candidates: List[FrozenSet[int]] = generate_subsets_extended(largest, m)

    solutions: List[Tuple[FrozenSet[int], ...]] = [] # Stores found solutions

    def backtrack(chosen_sets: List[FrozenSet[int]], start_idx: int):
        # 若已经选了 n 组
        if len(chosen_sets) == n:
            # chosen_sets 是 frozenset 的列表
            # 检查覆盖（当 nums 非空时）
            if nums: # nums 是 set
                union_set = set().union(*chosen_sets) # 计算并集
                if not nums.issubset(union_set):
                    return

                # --- Safety Check ---
                is_safe = True
                if nums: # Only check safety if head is not empty
                    for head_var in nums:
                        var_found_in_body = False
                        for body_set in chosen_sets:
                            if head_var in body_set:
                                var_found_in_body = True
                                break
                        if not var_found_in_body:
                            is_safe = False
                            break
                        else:
                            pass
                if not is_safe:
                    return # Discard unsafe solution
                else:
                    pass
                # --- Safety Check End ---


            # 检查连通性
            if is_connected(chosen_sets): # is_connected 接受 frozenset 列表
                solutions.append(tuple(chosen_sets)) # 添加解 (frozenset 的元组)
            else:
                pass
            return

        # 在 candidates 中继续选
        # 这里采用从 start_idx 往后遍历的方式，避免重复解太多
        # 如果想允许同一个子集出现多次，不要限制 start_idx，即改成 range(len(candidates))
        for i in range(start_idx, len(candidates)):
            s = candidates[i]
            chosen_sets.append(s)
            backtrack(chosen_sets, i)  # 若允许相同子集重复，则依旧从 i 开始
            chosen_sets.pop()

    # 初始状态：chosen_sets 是 frozenset 的列表
    chosen_sets = list(must_include_subsets) # must_include_subsets 是 frozenset 列表

    # 连通性检查在最后进行

    # 开始回溯，补足剩余 (n - len(chosen_sets)) 个子集
    backtrack(chosen_sets, 0)

    return solutions


# ============= Helper for calling cached solve =============
def solve_wrapper(nums: Set[int],
                  n: int,
                  m: int,
                  largest: int,
                  must_include_subsets: Optional[Iterable[Iterable[int]]] = None) -> List[Tuple[FrozenSet[int], ...]]:
    """
    Wrapper for the cached `solve` function. Converts mutable inputs (set, list of iterables)
    to hashable types (frozenset, tuple of frozensets) required for caching.

    Also performs validation and normalization on `must_include_subsets`.

    :param nums: A set of integers that must be covered.
    :type nums: Set[int]
    :param n: The exact number of subsets to choose.
    :type n: int
    :param m: The maximum size allowed for each chosen subset.
    :type m: int
    :param largest: The maximum integer value allowed within the subsets.
    :type largest: int
    :param must_include_subsets: An optional iterable (list, tuple) of iterables (list, tuple, set)
                                 representing subsets that must be included.
    :type must_include_subsets: Optional[Iterable[Iterable[int]]]
    :raises ValueError: If validation of `must_include_subsets` fails (count > n, invalid size, invalid elements).
    :return: A list of solutions from the cached `solve` function. Each solution is a tuple of frozensets.
    :rtype: List[Tuple[FrozenSet[int], ...]]
    """
    nums_frozen = frozenset(nums) if nums else frozenset()

    must_include_subsets_frozen: Optional[Tuple[FrozenSet[int], ...]] = None
    if must_include_subsets:
        # Convert each inner iterable to frozenset
        frozen_list: List[FrozenSet[int]] = [frozenset(s) for s in must_include_subsets]

        # --- Validation ---
        if len(frozen_list) > n:
            raise ValueError(f"Number of must-include subsets ({len(frozen_list)}) exceeds target n ({n}).")
        for fs in frozen_list:
            if not (1 <= len(fs) <= m):
                raise ValueError(f"Must-include subset {set(fs)} has invalid size {len(fs)}. Allowed range [1, {m}].")
            if any(not (0 <= x <= largest) for x in fs):
                raise ValueError(f"Must-include subset {set(fs)} contains elements outside range [0, {largest}].")
        # --- Validation End ---

        # Sort the list of frozensets based on their sorted element tuples to ensure canonical order for caching
        must_include_subsets_frozen = tuple(sorted(frozen_list, key=lambda fs: tuple(sorted(list(fs)))))

    return solve(nums_frozen, n, m, largest, must_include_subsets_frozen)


# ============= 测试示例 (using wrapper) =============
if __name__ == "__main__":
    # 情景1：nums={0,1,2}, 需要3组子集，每组大小<=6，largest=6
    # 固定子集 must_include_subsets=[(0,1), (0,2,3,4,5)]
    nums1 = {0, 1, 2}
    n1 = 3
    m1 = 6
    largest1 = 6
    must_include1 = [(0, 1), (0, 2, 3, 4, 5)]  # 固定子集

    res1 = solve_wrapper(nums1, n1, m1, largest1, must_include_subsets=must_include1)
    print(f"情景1: nums={nums1}, must_include_subsets={must_include1}")
    print(f"共找到 {len(res1)} 个解，示例展示前5个：")
    for sol in res1[:5]:
        # 解现在是 frozenset 的元组
        print(tuple(set(fs) for fs in sol)) # 打印时转回 set 元组方便查看

    print("========================================")

    # 情景2：nums={} (空集)，要3组，每组<=2个元素，largest=3
    # 固定子集 must_include_subsets=[(0,1), (1,2)]
    nums2 = set()
    n2 = 3
    m2 = 2
    largest2 = 3
    must_include2 = [(0, 1), (1, 2)]

    res2 = solve_wrapper(nums2, n2, m2, largest2, must_include_subsets=must_include2)
    print(f"情景2: nums={nums2}, must_include_subsets={must_include2}")
    print(f"共找到 {len(res2)} 个解，示例输出前5个：")
    for sol in res2[:5]:
        print(tuple(set(fs) for fs in sol)) # 打印时转回 set 元组

    # 测试缓存
    print("\nTesting cache:")
    res1_again = solve_wrapper(nums1, n1, m1, largest1, must_include_subsets=must_include1)
    print(f"情景1再次调用，找到 {len(res1_again)} 个解 (应与第一次相同)")
    # 验证缓存是否命中 (通常需要查看缓存信息，这里仅作演示)
    # print(solve.cache_info()) # 需要访问原始 solve 函数才能看 cache_info

    # 测试不同顺序的 must_include
    must_include1_reordered = [(0, 2, 3, 4, 5), (0, 1)]
    res1_reordered = solve_wrapper(nums1, n1, m1, largest1, must_include_subsets=must_include1_reordered)
    print(f"情景1使用重排的 must_include，找到 {len(res1_reordered)} 个解 (应与第一次相同，且缓存应命中)")
    # print(solve.cache_info())
