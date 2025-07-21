# Contains logic to build a networkx graph from crawled ConceptNet data
import networkx as nx
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_graph_from_edges(edges_data: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Builds a networkx DiGraph from a list of ConceptNet edge data.

    :param edges_data: A list of dictionaries, where each dictionary
                       represents a crawled edge and contains keys like
                       'start_uri', 'end_uri', 'relation_uri', 'weight'.
    :type edges_data: List[Dict[str, Any]]
    :return: A networkx directed graph representing the ConceptNet data.
    :rtype: nx.DiGraph
    """
    graph = nx.DiGraph()
    added_edges_count = 0

    for edge in edges_data:
        start_node = edge.get('start_uri')
        end_node = edge.get('end_uri')
        relation = edge.get('relation_uri')
        weight = edge.get('weight', 1.0) # Default weight if not present

        if not start_node or not end_node or not relation:
            logging.warning(f"Skipping edge due to missing data: {edge}")
            continue

        # Add nodes (networkx handles duplicates automatically)
        # Extract simple names for potential node labels if needed later
        start_name = start_node.split('/')[-1]
        end_name = end_node.split('/')[-1]
        graph.add_node(start_node, name=start_name, lang=start_node.split('/')[2])
        graph.add_node(end_node, name=end_name, lang=end_node.split('/')[2])

        # Add edge with attributes
        if not graph.has_edge(start_node, end_node, key=relation):
             graph.add_edge(start_node, end_node, key=relation, relation=relation, weight=weight)
             added_edges_count += 1
        else:
            # Optionally update weight if edge exists? For now, just skip duplicates.
            logging.debug(f"Edge {start_node} -[{relation}]-> {end_node} already exists.")


    logging.info(f"Built graph with {graph.number_of_nodes()} nodes and {added_edges_count} unique edges from scratch.")
    return graph


def update_graph_with_edges(graph: nx.DiGraph, new_edges_data: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Updates an existing networkx DiGraph with a list of new ConceptNet edge data.

    :param graph: The existing networkx graph object to update.
    :type graph: nx.DiGraph
    :param new_edges_data: A list of dictionaries representing newly crawled edges.
    :type new_edges_data: List[Dict[str, Any]]
    :return: The updated networkx graph object.
    :rtype: nx.DiGraph
    """
    if graph is None: # Should not happen if called correctly, but handle defensively
        logging.warning("Initial graph is None in update_graph_with_edges. Building new graph instead.")
        return build_graph_from_edges(new_edges_data)

    added_edges_count = 0
    initial_node_count = graph.number_of_nodes()
    initial_edge_count = graph.number_of_edges()

    for edge in new_edges_data:
        start_node = edge.get('start_uri')
        end_node = edge.get('end_uri')
        relation = edge.get('relation_uri')
        weight = edge.get('weight', 1.0)

        if not start_node or not end_node or not relation:
            logging.warning(f"Skipping new edge due to missing data: {edge}")
            continue

        # Add nodes if they don't exist
        start_name = start_node.split('/')[-1]
        end_name = end_node.split('/')[-1]
        if not graph.has_node(start_node):
            graph.add_node(start_node, name=start_name, lang=start_node.split('/')[2])
        if not graph.has_node(end_node):
            graph.add_node(end_node, name=end_name, lang=end_node.split('/')[2])

        # Add edge with attributes if it doesn't exist
        # Use relation as key for MultiDiGraph compatibility if needed later,
        # but DiGraph stores one edge per pair, so check needs key if attributes differ.
        # For simplicity with DiGraph, we check if the specific relation exists.
        # Note: NetworkX DiGraph doesn't directly support multiple edges between
        # the same two nodes easily without MultiDiGraph. We assume one edge per relation type.
        if not graph.has_edge(start_node, end_node) or graph[start_node][end_node].get('relation') != relation:
             # If using MultiDiGraph, use add_edge(..., key=relation, ...)
             # For DiGraph, this might overwrite if an edge exists but with a different relation stored in attrs.
             # Let's assume we want to store the relation type in the attributes.
             graph.add_edge(start_node, end_node, relation=relation, weight=weight)
             added_edges_count += 1
        else:
             logging.debug(f"Edge {start_node} -[{relation}]-> {end_node} already exists in graph.")

    final_node_count = graph.number_of_nodes()
    final_edge_count = graph.number_of_edges() # Note: This counts total edges, not just added ones.

    logging.info(f"Updated graph. Added {final_node_count - initial_node_count} nodes and {added_edges_count} new unique edges.")
    logging.info(f"Graph now has {final_node_count} nodes and {final_edge_count} total edges.")
    return graph
