from collections import deque

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# import matplotlib # No longer needed just for colormaps

# Explicitly set default font to support Chinese characters and avoid issues
# Provide a list of potential fonts; matplotlib will use the first one found.
# Removed 'SimHei' as it was causing warnings. Trying 'SimSun' as a common fallback.
# plt.rcParams['font.sans-serif'] = ['SimSun', 'sans-serif'] # Commented out as SimSun also not found
plt.rcParams['axes.unicode_minus'] = False # Keep this for minus sign display

def topological_sort(adj_matrix: np.ndarray) -> list[int]:
    """
    Perform topological sort on the given DAG.
    Returns a list of nodes such that for every directed edge u->v, u comes before v in the ordering.
    """
    num_nodes = len(adj_matrix)
    in_degree = [0]*num_nodes
    for u in range(num_nodes):
        for v in range(num_nodes):
            if adj_matrix[u][v] == 1:
                in_degree[v] += 1
    # Kahn 算法
    queue = deque()
    for i in range(num_nodes):
        if in_degree[i] == 0:
            queue.append(i)
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in range(num_nodes):
            if adj_matrix[u][v] == 1:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
    # 如果是无环图，topo_order 的长度应当等于 num_nodes
    # 若有环的话，这里长度会小于 num_nodes
    if len(topo_order) < num_nodes:
        raise ValueError("Graph contains a cycle, cannot perform topological sort!")
    return topo_order


def check_path_exists(adj_matrix: np.ndarray, start_node: int, end_node: int) -> bool:
    """
    检查在给定的邻接矩阵表示的有向图中，是否存在从 start_node 到 end_node 的路径。
    使用深度优先搜索 (DFS)。

    :param adj_matrix: 邻接矩阵 (numpy array)，adj_matrix[i, j] > 0 表示存在边 i->j。
    :param start_node: 起始节点索引。
    :param end_node: 目标节点索引。
    :return: 如果存在路径，返回 True；否则返回 False。
    """
    num_nodes = adj_matrix.shape[0]
    if not (0 <= start_node < num_nodes and 0 <= end_node < num_nodes):
        # 如果起始或结束节点无效，则路径不存在
        return False
    if start_node == end_node:
        # 如果起点和终点相同，则路径存在（长度为0）
        return True

    visited = {start_node}
    stack = [start_node]

    while stack:
        current_node = stack.pop()
        # 查找当前节点的所有邻居
        neighbors = np.where(adj_matrix[current_node, :] > 0)[0]
        for neighbor in neighbors:
            if neighbor == end_node:
                return True # 找到路径
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    return False # 遍历完成，未找到路径


def visualize_heterogeneous_digraph(adjacency_matrix: np.ndarray, idx2type:dict[int, str]=None) -> None:
    """
    Visualize a heterogeneous directed graph using NetworkX + Matplotlib.
    Nodes are assigned different shapes and colors based on their 'type'.
    Edges are assigned different colors and styles based on their value in the adjacency matrix.

    Args:
        adjacency_matrix: numpy.ndarray or list of lists
          N×N matrix where adjacency_matrix[i][j] != 0 indicates a directed edge i->j,
          and its value represents the edge type.
        idx2type: dict, optional
          Mapping {node_id: node_type}. Node types are strings.
          If None or empty, all nodes are treated as the same type.

    Note:
        1. You can extend this for more node/edge types; colors/shapes/styles will cycle if types exceed defaults.
    """
    adjacency_matrix = np.asarray(adjacency_matrix) # Ensure it's a numpy array
    n = adjacency_matrix.shape[0]

    # 1) 处理节点类型
    if not idx2type:
        idx2type = {i: 'Default' for i in range(n)}

    # 2) 创建有向图
    G = nx.DiGraph()

    # 3) 添加节点
    for node_id, node_type in idx2type.items():
        if 0 <= node_id < n: # Ensure node_id is valid
             G.add_node(node_id, type=node_type)

    # 4) 添加边并记录类型
    unique_edge_types_set = set()
    for i in range(n):
        for j in range(n):
            edge_value = adjacency_matrix[i][j]
            if edge_value != 0:
                # Ensure nodes exist before adding edge
                if G.has_node(i) and G.has_node(j):
                    G.add_edge(i, j, type=edge_value)
                    unique_edge_types_set.add(edge_value)

    # 5) 准备可视化布局
    plt.figure(figsize=(10, 8), facecolor='whitesmoke') # Increased size slightly
    try:
      # Use seed for reproducibility, k for spacing
      pos = nx.spring_layout(G, seed=42, k=0.8/np.sqrt(len(G.nodes())) if len(G.nodes()) > 0 else 0.8)
    except nx.NetworkXException: # Handle case with no nodes/edges
        print("Graph is empty or layout cannot be computed.")
        plt.close()
        return

    # --- 节点样式 ---
    # 6) Collect node types, assign colors and shapes
    unique_node_types = sorted(list(set(idx2type.values())))
    node_color_map = {}
    node_shape_map = {}
    # Ensure get_cmap has enough types, even if 0
    num_node_types_for_cmap = max(1, len(unique_node_types))
    # Get the colormap object first
    node_cmap = plt.colormaps.get_cmap("Pastel1")
    shape_candidates = ['o', '^', 's', 'D', 'v', '>', '<', 'p', 'h']
    for i, t in enumerate(unique_node_types):
        # Calculate color based on index and total types
        node_color_map[t] = node_cmap(i / num_node_types_for_cmap if num_node_types_for_cmap > 1 else 0.5) # Normalize index for cmap
        node_shape_map[t] = shape_candidates[i % len(shape_candidates)]

    # 7) 将节点根据类型分组
    type2nodes = {}
    for node in G.nodes():
        # Ensure node has 'type' attribute before accessing
        if 'type' in G.nodes[node]:
            t = G.nodes[node]['type']
            if t not in type2nodes:
                type2nodes[t] = []
            type2nodes[t].append(node)
        # else: handle nodes potentially added without type? (Shouldn't happen with current logic)

    # --- 边样式 ---
    # 8) 统计边类型，分配颜色、样式和箭头样式
    unique_edge_types = sorted(list(unique_edge_types_set))
    edge_color_map = {}
    edge_style_map = {}
    arrowstyle_map = {} # Arrow style map
    if unique_edge_types: # Only create maps if edges exist
        # Ensure get_cmap has enough types, even if 0
        num_edge_types_for_cmap = max(1, len(unique_edge_types))
        # Get the colormap object first
        edge_cmap = plt.colormaps.get_cmap("viridis") # Different cmap for edges
        style_candidates = ['-', '--', ':', '-.']
        # 仅使用指向前方的箭头样式
        arrowstyle_candidates = ['-|>', '-|>', '->', '->'] # Forward pointing arrow styles only
        for i, etype in enumerate(unique_edge_types):
            # Calculate color based on index and total types
            edge_color_map[etype] = edge_cmap(i / num_edge_types_for_cmap if num_edge_types_for_cmap > 1 else 0.5) # Normalize index for cmap
            edge_style_map[etype] = style_candidates[i % len(style_candidates)]
            # 使用调整后的候选列表分配箭头样式
            arrowstyle_map[etype] = arrowstyle_candidates[i % len(arrowstyle_candidates)]

    # --- 绘图 ---
    # 9) 绘制节点 (分类型)
    node_handles = []
    for node_type, nodelist in type2nodes.items():
         # Check if nodelist is not empty to avoid errors with plt.scatter
        if nodelist:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodelist,
                node_color=[node_color_map[node_type]] * len(nodelist),
                node_shape=node_shape_map[node_type],
                # label=f"Node: {node_type}", # Labeling nodes directly in legend can be crowded
                node_size=700,
                linewidths=1.0,
                edgecolors='black',
                alpha=0.9
            )
            # Create proxy artist for legend (English label)
            node_handles.append(plt.scatter([],[], color=node_color_map[node_type], marker=node_shape_map[node_type], label=f"Node Type: {node_type}"))


    # 10) 绘制边 (分类型)
    edge_handles = []
    if unique_edge_types: # Only draw edges if they exist
        for edge_type in unique_edge_types:
            edgelist = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == edge_type]
            if edgelist: # Only draw if edges of this type exist
                 nx.draw_networkx_edges(
                    G, pos,
                    edgelist=edgelist,
                    edge_color=[edge_color_map[edge_type]] * len(edgelist),
                    style=edge_style_map[edge_type],
                    arrows=True,
                    arrowstyle=arrowstyle_map[edge_type], # 修改：使用映射的箭头样式
                    arrowsize=25,
                    width=1.5,
                    node_size=700 # Match node size to avoid overlap
                )
                 # Create proxy artist for legend (English label, legend doesn't easily show arrow style)
                 edge_handles.append(plt.plot([],[], color=edge_color_map[edge_type], linestyle=edge_style_map[edge_type], label=f"Edge Type: {edge_type}")[0])


    # 11) 绘制标签
    nx.draw_networkx_labels(G, pos, font_color='black', font_size=9, font_weight='normal')

    # 12) 显示图例和绘图
    if node_handles or edge_handles: # Only show legend if there's something to label
        plt.legend(handles=node_handles + edge_handles, loc='best', fontsize='small')
    plt.title("Heterogeneous Directed Graph Visualization (Node Types + Edge Types)", fontsize=14)
    plt.axis('off')
    plt.tight_layout() # Adjust layout
    plt.show()


def longest_path_to_target(adj:np.ndarray, t:int) -> dict[int, int]:
    """
    计算在有向无环图 adj 中，从每个节点到目标节点 t 的最长路径长度。
    如果某节点无法到达 t，则其 dist 值会保持为 -inf。

    参数:
      adj: 邻接矩阵 (list[list[int]]), adj[i][j] = 1 表示 i → j 有向边；0 表示无边
      t:   目标节点编号
    返回:
      dist: 长度为 n 的列表, dist[i] 即为从 i 到 t 的最长路径长度
    """
    import sys
    sys.setrecursionlimit(10 ** 7)
    n = len(adj)

    # 1) 构造反转图：若原图中有 i→j，则反转图中有 j→i
    rev_adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j] == 1:
                rev_adj[j][i] = 1

    # 2) 对反转图进行拓扑排序
    visited = [False] * n
    topo = []

    def dfs(u):
        visited[u] = True
        for v in range(n):
            # 如果 rev_adj[u][v] == 1，表示反转图中有边 u→v (原图 v→u)
            if rev_adj[u][v] == 1 and not visited[v]:
                dfs(v)
        topo.append(u)

    for i in range(n):
        if not visited[i]:
            dfs(i)
    topo.reverse()  # 得到拓扑序

    # 3) 在反转图中，以 t 为起点，做“最长路径”DP
    dist = [-float('inf')] * n
    dist[t] = 0  # 到达目标节点 t 自身的距离为 0

    for u in topo:
        # 如果 dist[u] 还没更新过，表示在反转图中从 t 到 u 不可达(原图中即 u→t 不通)
        if dist[u] == -float('inf'):
            continue
        for v in range(n):
            if rev_adj[u][v] == 1:
                # 在反转图中有 u→v ⇒ 原图中 v→u
                if dist[u] + 1 > dist[v]:
                    dist[v] = dist[u] + 1

    return dist
