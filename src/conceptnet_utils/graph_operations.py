# -*- coding: utf-8 -*-
"""
Contains functions for operating on a loaded ConceptNet graph (nx.DiGraph).
"""
import networkx as nx
import random
import logging
from typing import Optional, List, Dict, Any, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_random_node(graph: nx.DiGraph) -> Optional[str]:
    """
    Selects a random node URI from the graph.

    :param graph: The networkx DiGraph object representing the ConceptNet data.
    :type graph: nx.DiGraph
    :return: A randomly selected node URI (string), or None if the graph is empty.
    :rtype: Optional[str]
    """
    if not graph:
        logging.warning("Graph is empty or None. Cannot select a random node.")
        return None
    nodes = list(graph.nodes())
    if not nodes:
        logging.warning("Graph has no nodes. Cannot select a random node.")
        return None
    return random.choice(nodes)

def get_connected_info(graph: nx.DiGraph, node_uri: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Retrieves information about nodes and edges connected to a specific node URI.
    It separates connections into 'outgoing' (successors) and 'incoming' (predecessors).

    :param graph: The networkx DiGraph object.
    :type graph: nx.DiGraph
    :param node_uri: The URI of the node to get connection info for.
    :type node_uri: str
    :return: A dictionary with keys 'outgoing' and 'incoming'. Each key holds a list
             of dictionaries, where each dictionary describes a connection (neighbor node
             and edge data). Returns None if the node_uri is not found in the graph.
    :rtype: Optional[Dict[str, List[Dict[str, Any]]]]
    """
    if not graph.has_node(node_uri):
        logging.error(f"Node URI '{node_uri}' not found in the graph.")
        return None

    connected_info: Dict[str, List[Dict[str, Any]]] = {
        "outgoing": [],
        "incoming": []
    }

    # --- Process Outgoing Edges (Successors) ---
    for successor_uri in graph.successors(node_uri):
        # NetworkX DiGraph stores edge data in graph[u][v]
        edge_data = graph.get_edge_data(node_uri, successor_uri)
        # Get neighbor node data
        neighbor_node_data = graph.nodes[successor_uri]

        connection = {
            "direction": "outgoing",
            "neighbor_node": {
                "uri": successor_uri,
                "name": neighbor_node_data.get("name", successor_uri.split('/')[-1]), # Default to name from URI if 'name' attr missing
                **neighbor_node_data # Include all other node attributes
            },
            "edge_data": {
                **edge_data # Include all edge attributes (relation, weight, surfaceText, etc.)
            }
        }
        connected_info["outgoing"].append(connection)
        logging.debug(f"Found outgoing connection: {node_uri} -> {successor_uri}")

    # --- Process Incoming Edges (Predecessors) ---
    for predecessor_uri in graph.predecessors(node_uri):
        # Edge data is stored based on the edge direction (predecessor -> node_uri)
        edge_data = graph.get_edge_data(predecessor_uri, node_uri)
        # Get neighbor node data
        neighbor_node_data = graph.nodes[predecessor_uri]

        connection = {
            "direction": "incoming",
            "neighbor_node": {
                "uri": predecessor_uri,
                "name": neighbor_node_data.get("name", predecessor_uri.split('/')[-1]),
                **neighbor_node_data
            },
            "edge_data": {
                **edge_data
            }
        }
        connected_info["incoming"].append(connection)
        logging.debug(f"Found incoming connection: {predecessor_uri} -> {node_uri}")

    logging.info(f"Retrieved connection info for node '{node_uri}': "
                 f"{len(connected_info['outgoing'])} outgoing, "
                 f"{len(connected_info['incoming'])} incoming.")

    return connected_info

# Example Usage (can be run if this file is executed directly)
if __name__ == '__main__':
    # Create a sample graph for testing
    G = nx.DiGraph()
    G.add_node("/c/en/cat", name="cat", lang="en")
    G.add_node("/c/en/animal", name="animal", lang="en")
    G.add_node("/c/en/pet", name="pet", lang="en")
    G.add_node("/c/en/dog", name="dog", lang="en")
    G.add_node("/c/en/mammal", name="mammal", lang="en")

    G.add_edge("/c/en/cat", "/c/en/animal", relation="/r/IsA", weight=2.0, surfaceText="[[cat]] is a type of [[animal]]")
    G.add_edge("/c/en/cat", "/c/en/pet", relation="/r/UsedFor", weight=1.5, surfaceText="[[cat]] can be a [[pet]]")
    G.add_edge("/c/en/animal", "/c/en/mammal", relation="/r/IsA", weight=2.0, surfaceText="[[animal]] is a type of [[mammal]]")
    G.add_edge("/c/en/dog", "/c/en/animal", relation="/r/IsA", weight=2.0, surfaceText="[[dog]] is a type of [[animal]]")
    G.add_edge("/c/en/pet", "/c/en/animal", relation="/r/IsA", weight=1.0, surfaceText="a [[pet]] is an [[animal]]") # Incoming to animal

    print("--- Testing get_random_node ---")
    random_node = get_random_node(G)
    print(f"Randomly selected node: {random_node}")
    random_node = get_random_node(G)
    print(f"Randomly selected node: {random_node}")

    print("\n--- Testing get_connected_info for /c/en/cat ---")
    cat_info = get_connected_info(G, "/c/en/cat")
    if cat_info:
        import json
        print(json.dumps(cat_info, indent=2))

    print("\n--- Testing get_connected_info for /c/en/animal ---")
    animal_info = get_connected_info(G, "/c/en/animal")
    if animal_info:
        import json
        print(json.dumps(animal_info, indent=2))

    print("\n--- Testing get_connected_info for non-existent node ---")
    non_existent_info = get_connected_info(G, "/c/en/bird")
    print(f"Info for non-existent node: {non_existent_info}")

    print("\n--- Testing with empty graph ---")
    empty_graph = nx.DiGraph()
    random_empty = get_random_node(empty_graph)
    print(f"Random node from empty graph: {random_empty}")
    info_empty = get_connected_info(empty_graph, "/c/en/cat")
    print(f"Info from empty graph: {info_empty}")
