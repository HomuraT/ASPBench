# Contains functions for saving and loading the networkx graph
# Contains functions for saving and loading the networkx graph
import networkx as nx
import logging
import os
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_graph_to_graphml(graph: nx.DiGraph, filepath: str) -> None:
    """
    Saves a networkx graph to a GraphML file.

    :param graph: The networkx graph object to save.
    :type graph: nx.DiGraph
    :param filepath: The path to the output GraphML file.
    :type filepath: str
    :raises Exception: Re-raises any exception encountered during saving after logging.
    :return: None
    :rtype: None
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nx.write_graphml(graph, filepath)
        logging.info(f"Graph successfully saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving graph to {filepath}: {e}")
        raise # Re-raise the exception after logging

def load_graph_from_graphml(filepath: str) -> Optional[nx.DiGraph]:
    """
    Loads a networkx graph from a GraphML file.

    :param filepath: The path to the input GraphML file.
    :type filepath: str
    :return: The loaded networkx graph object, or None if loading fails.
    :rtype: Optional[nx.DiGraph]
    """
    if not os.path.exists(filepath):
        logging.error(f"Graph file not found: {filepath}")
        return None
    try:
        graph = nx.read_graphml(filepath)
        logging.info(f"Graph successfully loaded from {filepath}")
        return graph
    except Exception as e:
        logging.error(f"Error loading graph from {filepath}: {e}")
        return None
