import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
from src.utils.sparse_utils import sparse_serializable_to_dense, dense_to_sparse_serializable # Import the conversion utilities

# Define constants for clarity based on the latest interpretation
# Rule -> Predicate (Head)
POS_HEAD_TYPES = {1, 7}
NEG_HEAD_TYPES = {3, 5} # Assuming strong negation (~h)

# Predicate -> Rule (Body) - Assuming 1=pos, 3=strong_neg, 5=default_neg
# And assuming 7 = 1 | 3 | 5 (bitwise OR, meaning all types present)
# If 7 has a different meaning, the body parsing needs adjustment.
BODY_POS_MASK = 1
BODY_STRONG_NEG_MASK = 3 # Using 3 directly as the mask/value
BODY_DEFAULT_NEG_MASK = 5 # Using 5 directly as the mask/value
BODY_COMBINED_MASK = 7 # Represents the presence of 1, 3, and 5? Or just value 7?
                       # Let's treat 1, 3, 5, 7 as distinct values for now,
                       # aligning with the idea that 7 might be its own type
                       # or simply includes 1, 3, 5. The bitwise check covers this.

# Dependency Types in the predicate dependency graph
DEP_POS = 1
DEP_STRONG_NEG = 3 # Dependency introduced by strong negation
DEP_DEFAULT_NEG = 5 # Dependency introduced by default negation


class ASPProgramAnalyzer:
    """
    Analyzes properties of an ASP program represented by a sparse adjacency matrix.

    Assumes the graph representation conventions:
    - Nodes are predicates or rules.
    - Edge `Rule -> Predicate` (adj[rule, pred]): Represents the head.
        - data = 1 or 7: Positive head (h)
        - data = 3 or 5: Negative head (~h, strong negation assumed)
    - Edge `Predicate -> Rule` (adj[pred, rule]): Represents the body.
        - data = 1: Positive body atom (p)
        - data = 3: Strong negative body atom (~p)
        - data = 5: Default negative body atom (not p)
        - data = 7: Assumed to potentially indicate multiple roles (p, ~p, not p)
                     Handled by checking specific bits/values if needed, but
                     we will treat 1, 3, 5, 7 as distinct indicators for now.
                     More precisely, we check for 1, 3, 5 directly.
    """
    def __init__(self, adj_sparse_dict: Dict[str, Any], idx_to_type: Dict[int, str], use_disjunction: bool = False):
        """
        Initializes the ASPProgramAnalyzer with a sparse adjacency matrix dictionary.

        Args:
            adj_sparse_dict: A serializable dictionary representation of the
                             sparse adjacency matrix, as produced by
                             dense_to_sparse_serializable.
            idx_to_type: A dictionary mapping node indices to their type ('predicate' or 'rule').
            use_disjunction (bool): If True, treat disjunctions in heads as
                                    single rules. If False, decompose rules
                                    with multiple heads into multiple rules
                                    with single heads. Defaults to False.
        """
        self.adj_sparse_dict = adj_sparse_dict
        self.use_disjunction = use_disjunction
        # We will convert to dense matrix only when needed within methods
        # For initialization, we only need the shape which is in the dict
        self.num_nodes = adj_sparse_dict['shape'][0]
        self.idx_to_type = idx_to_type

        self.predicate_indices = {
            idx for idx, type_name in idx_to_type.items()
            if type_name == 'predicate'
        }
        self.original_rule_indices = {
            idx for idx, type_name in idx_to_type.items()
            if type_name == 'rule'
        }

        # Precompute rule details and dependency graph for efficiency
        # _get_all_rule_details will handle decomposition based on use_disjunction
        self.rule_details = self._get_all_rule_details()
        # The set of 'rule' indices used in subsequent steps will depend on decomposition
        self.rule_indices = set(self.rule_details.keys())

        self.predicate_dependency_graph = self._build_predicate_dependency_graph()
        self.predicate_dep_nx_graph = self._build_nx_dependency_graph()


    def _get_rule_details(self, r_idx: int) -> Dict[str, Set[int]]:
        """
        Extracts head and body predicates for a single rule.

        Args:
            r_idx (int): The index of the rule.

        Returns:
            Dict[str, Set[int]]: A dictionary containing sets of head and body
                                 predicate indices categorized by type.
        """
        # Convert to dense matrix for easier slicing/indexing in this method
        adj_matrix_dense = sparse_serializable_to_dense(self.adj_sparse_dict)

        details = {
            "head_pos": set(),
            "head_neg": set(),
            "body_pos": set(),
            "body_strong_neg": set(),
            "body_default_neg": set(),
        }

        # Find heads
        for p_idx in self.predicate_indices:
            p_idx = int(p_idx)
            head_type = adj_matrix_dense[r_idx, p_idx]
            if head_type in POS_HEAD_TYPES:
                details["head_pos"].add(p_idx)
            elif head_type in NEG_HEAD_TYPES:
                details["head_neg"].add(p_idx)

        # Find bodies
        for p_idx in self.predicate_indices:
            p_idx = int(p_idx)
            body_type = adj_matrix_dense[p_idx, r_idx]
            # Check for specific body types. Assumes 7 doesn't exclusively mean
            # something else, but includes the basic types.
            if body_type == 1:
                 details["body_pos"].add(p_idx)
            elif body_type == 3:
                 details["body_strong_neg"].add(p_idx)
            elif body_type == 5:
                 details["body_default_neg"].add(p_idx)
            elif body_type == 7:
                # Reverting to direct checks as bitmasks 3 and 5 are problematic:
                if body_type == 1: details["body_pos"].add(p_idx)
                elif body_type == 3: details["body_strong_neg"].add(p_idx)
                elif body_type == 5: details["body_default_neg"].add(p_idx)
                elif body_type == 7:
                    # What does 7 mean for the body? Let's assume it's like 1 (positive). Needs confirmation.
                    # print(f"Warning: Body edge type 7 from P{p_idx} to R{r_idx} encountered. Interpretation unclear. Treating as positive.")
                    details["body_pos"].add(p_idx)


        return details

    def _get_all_rule_details(self) -> Dict[int, Dict[str, Set[int]]]:
        """
        Computes details for all rules, handling disjunction based on
        self.use_disjunction.

        If use_disjunction is False, rules with multiple heads are decomposed
        into multiple single-head rules. New virtual rule indices are created
        for these decomposed rules.
        """
        all_details = {}
        virtual_rule_idx_counter = self.num_nodes # Start counter after original nodes

        # Convert to dense matrix only once here for efficiency
        adj_matrix_dense = sparse_serializable_to_dense(self.adj_sparse_dict)

        for r_idx in self.original_rule_indices:
            r_idx = int(r_idx)
            # Extract head and body details for the original rule
            details = {
                "head_pos": set(),
                "head_neg": set(),
                "body_pos": set(),
                "body_strong_neg": set(),
                "body_default_neg": set(),
            }

            # Find heads for the original rule
            for p_idx in self.predicate_indices:
                p_idx = int(p_idx)
                head_type = adj_matrix_dense[r_idx, p_idx]
                if head_type in POS_HEAD_TYPES:
                    details["head_pos"].add(p_idx)
                elif head_type in NEG_HEAD_TYPES:
                    details["head_neg"].add(p_idx)

            # Find bodies for the original rule (body is the same for decomposed rules)
            for p_idx in self.predicate_indices:
                p_idx = int(p_idx)
                body_type = adj_matrix_dense[p_idx, r_idx]
                if body_type == 1:
                     details["body_pos"].add(p_idx)
                elif body_type == 3:
                     details["body_strong_neg"].add(p_idx)
                elif body_type == 5:
                     details["body_default_neg"].add(p_idx)
                elif body_type == 7:
                    # Assuming 7 means positive body atom based on prior note
                    details["body_pos"].add(p_idx)


            if self.use_disjunction or (len(details["head_pos"]) + len(details["head_neg"])) <= 1:
                # If using disjunction or rule has 0 or 1 head, keep as is
                all_details[r_idx] = details
            else:
                # If not using disjunction and rule has multiple heads, decompose
                original_body_details = {k: v.copy() for k, v in details.items() if k.startswith("body_")}
                # Decompose positive heads
                for p_head in details["head_pos"]:
                    decomposed_details = original_body_details.copy()
                    decomposed_details["head_pos"] = {p_head}
                    decomposed_details["head_neg"] = set() # No negative head in this decomposed rule
                    all_details[virtual_rule_idx_counter] = decomposed_details
                    virtual_rule_idx_counter += 1
                # Decompose negative heads
                for p_head in details["head_neg"]:
                     decomposed_details = original_body_details.copy()
                     decomposed_details["head_pos"] = set() # No positive head in this decomposed rule
                     decomposed_details["head_neg"] = {p_head}
                     all_details[virtual_rule_idx_counter] = decomposed_details
                     virtual_rule_idx_counter += 1

        return all_details

    def _build_predicate_dependency_graph(self) -> Dict[int, Set[Tuple[int, int]]]:
        """
        Builds the dependency graph between predicates based on the (potentially decomposed)
        rule details.
        Stores dependencies as: {body_pred_idx: {(head_pred_idx, dep_type), ...}}
        dep_type: DEP_POS, DEP_STRONG_NEG, DEP_DEFAULT_NEG
        """
        graph = defaultdict(set)
        # Iterate over potentially virtual rule indices from self.rule_details
        for r_idx, details in self.rule_details.items():
            head_pos = details["head_pos"]
            head_neg = details["head_neg"]

            # Positive body dependencies
            for p_body in details["body_pos"]:
                for h_pos in head_pos:
                    graph[p_body].add((h_pos, DEP_POS))
                for h_neg in head_neg:
                    # p -> ~h : negative dependency (strong neg type)
                    graph[p_body].add((h_neg, DEP_STRONG_NEG))

            # Strong negative body dependencies
            for p_body in details["body_strong_neg"]:
                for h_pos in head_pos:
                    # ~p -> h : negative dependency (strong neg type)
                    graph[p_body].add((h_pos, DEP_STRONG_NEG))
                for h_neg in head_neg:
                    # ~p -> ~h : positive dependency
                    graph[p_body].add((h_neg, DEP_POS))

            # Default negative body dependencies
            for p_body in details["body_default_neg"]:
                for h_pos in head_pos:
                    # not p -> h : negative dependency (default neg type)
                    graph[p_body].add((h_pos, DEP_DEFAULT_NEG))
                for h_neg in head_neg:
                    # not p -> ~h : negative dependency (default neg type)
                    # This was previously incorrect (DEP_POS), should be default neg
                    graph[p_body].add((h_neg, DEP_DEFAULT_NEG))

        # Ensure all predicates exist as keys, even if they don't appear in bodies
        for p_idx in self.predicate_indices:
            if p_idx not in graph:
                graph[p_idx] = set()

        return dict(graph) # Convert back to regular dict if needed

    def _build_nx_dependency_graph(self) -> nx.DiGraph:
        """
        Builds a NetworkX DiGraph from the predicate dependency graph.
        Note: This graph is built from the computed dependencies, which implicitly
        accounts for rule decomposition if use_disjunction was False.
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.predicate_indices)
        for body_pred, dependencies in self.predicate_dependency_graph.items():
            for head_pred, dep_type in dependencies:
                # Add edge from body to head, store type as attribute
                G.add_edge(body_pred, head_pred, type=dep_type)
        return G

    # --- Classification Methods ---


    def is_positive(self) -> bool:
        """Checks if the program is positive (no negations in bodies)."""
        # Note: This definition usually considers *only* default negation (not p).
        # If strong negation (~p) also makes it non-positive, adjust the check.
        # Let's assume standard definition: only `not p` makes it non-positive.
        for dependencies in self.predicate_dependency_graph.values():
            for _, dep_type in dependencies:
                if dep_type == DEP_DEFAULT_NEG:
                    return False
                # If strong negation also counts:
                # if dep_type in {DEP_DEFAULT_NEG, DEP_STRONG_NEG}:
                #     return False
        return True

    def is_tight(self) -> bool:
        """
        Checks if the program is tight (Head-Cycle-Free).
        A program is tight if its "positive dependency graph" is acyclic.
        The positive dependency graph includes dependencies derived from
        positive body atoms (p -> h, p -> ~h) and strongly negated
        body atoms (~p -> h, ~p -> ~h). It excludes dependencies
        derived from default negation (not p).
        Corresponds to DEP_POS and DEP_STRONG_NEG edge types in our graph.
        """
        # Create a graph with only "positive" dependencies (DEP_POS and DEP_STRONG_NEG)
        positive_dep_graph = nx.DiGraph()
        positive_dep_graph.add_nodes_from(self.predicate_indices)

        for u, v, data in self.predicate_dep_nx_graph.edges(data=True):
            # Include edges representing positive or strong negative dependencies
            if data.get('type') in {DEP_POS, DEP_STRONG_NEG}:
                positive_dep_graph.add_edge(u, v)

        # Check for cycles in this specific positive dependency graph
        try:
            nx.find_cycle(positive_dep_graph, orientation='original')
            return False # Cycle found, so not tight
        except nx.NetworkXNoCycle:
            return True # No cycle found, so tight


    def is_stratified(self) -> bool:
        """
        Checks if the program is stratified (no recursion through default negation).
        This implementation assumes standard stratification (often for non-disjunctive).
        It checks for cycles in the dependency graph involving default negation.
        Strong negation typically does not break stratification.
        """
        try:
            # Get all cycles in the full dependency graph
            cycles = list(nx.simple_cycles(self.predicate_dep_nx_graph))
        except Exception as e:
            # Handle potential errors during cycle finding if necessary
             print(f"Error finding cycles for stratification check: {e}")
             # Depending on the desired behavior, you might return False or raise
             return False # Assume non-stratified on error

        # Check if any cycle contains an edge representing default negation
        for cycle in cycles:
            # Need to check edges within the cycle
            edges_in_cycle = list(zip(cycle, cycle[1:] + cycle[:1]))
            for u, v in edges_in_cycle:
                if self.predicate_dep_nx_graph.has_edge(u, v):
                   edge_data = self.predicate_dep_nx_graph.get_edge_data(u, v)
                   if edge_data.get('type') == DEP_DEFAULT_NEG:
                       return False # Found a cycle through default negation

        return True # No cycles through default negation found

    def is_disjunctive_stratified(self) -> bool:
        """
        Checks if the program is disjunctively stratified.
        This is a more complex property. A common definition involves checking
        the component graph for negative cycles (where negative means involving
        default negation between components or within a component).

        This implementation checks for cycles involving default negation edges.
        A more rigorous check might involve explicit level mapping or
        component graph analysis, which can be quite involved. This simplified
        check covers the most common reason for non-stratification.
        """
        # For many practical purposes, the standard stratification check
        # (no cycles through default negation) is a strong indicator.
        # A truly precise check for *disjunctive* stratification can vary
        # based on specific semantics (e.g., DLP, DLV).
        # Let's reuse the standard stratification check as a proxy.
        # If a program is stratified, it's often considered disjunctively stratified
        # under some definitions, though not all.
        # print("Warning: is_disjunctive_stratified check is using the standard "
        #       "stratification check (no cycles through default negation). "
        #       "Precise disjunctive stratification definitions can be more complex.")
        return self.is_stratified()

    def classify(self) -> Dict[str, bool]:
        """
        Runs all classification checks and returns the results.

        Returns the result of is_disjunctive_stratified if use_disjunction is
        True, otherwise returns the result of is_stratified.
        """
        classification_results = {
            "is_positive": self.is_positive(),
            "is_tight": self.is_tight(),
            # "is_head_cycle_free" is now merged into is_tight
        }

        if self.use_disjunction:
            classification_results["is_disjunctive_stratified"] = self.is_disjunctive_stratified()
        else:
            classification_results["is_stratified"] = self.is_stratified()

        return classification_results
