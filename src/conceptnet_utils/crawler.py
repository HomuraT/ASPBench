
# Contains the logic for crawling ConceptNet API using synchronous requests
import requests
import logging
import time
from typing import Optional, Set, Dict, Any, Callable
import networkx as nx
from collections import deque
# tqdm might be used in the calling script, not directly here unless needed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONCEPTNET_API_URL = "http://api.conceptnet.io"
REQUEST_TIMEOUT = 30 # Seconds
# How often to check for saving (in number of edges added)
# SAVE_CHECK_INTERVAL = 50 # No longer needed for simple sync version
# Add a delay between requests to be polite to the API
REQUEST_DELAY = 0.5 # Seconds

def fetch_conceptnet_data_sync(uri: str) -> Optional[Dict[str, Any]]:
    """
    Synchronously fetches data for a given ConceptNet URI using requests.

    :param uri: The ConceptNet URI suffix (e.g., '/c/en/bird').
    :type uri: str
    :return: The JSON response as a dictionary, or None if an error occurs.
    :rtype: Optional[Dict[str, Any]]
    """
    full_url = f"{CONCEPTNET_API_URL}{uri}"
    logging.debug(f"Fetching URL: {full_url}")
    try:
        # Use requests.get, allow redirects by default
        response = requests.get(full_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        logging.debug(f"Successfully fetched data for {uri}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Requests error fetching {full_url}: {e}")
        return None
    except requests.exceptions.JSONDecodeError as e:
        logging.error(f"JSON decode error fetching {full_url}: {e}")
        return None
    except Exception as e: # Catch other potential errors
        logging.error(f"Unexpected error fetching {full_url}: {e}")
        return None

def crawl_conceptnet_sync(
    graph: nx.DiGraph, # Graph to modify in-place
    start_node_uri: str,
    allowed_relations: Set[str],
    target_total_edges: int, # Target total number of edges in the graph
    save_callback: Optional[Callable[[nx.DiGraph], None]] = None, # Synchronous save callback
    save_interval_edges: int = 100,
    progress_callback: Optional[Callable[[int], None]] = None # Progress callback
) -> int:
    """
    Synchronously crawls ConceptNet and updates the provided graph object in-place.
    Uses requests and a simple loop with a deque.

    :param graph: The networkx DiGraph object to update. It will be modified directly.
    :type graph: nx.DiGraph
    :param start_node_uri: The starting ConceptNet URI (e.g., '/c/en/bird').
    :type start_node_uri: str
    :param allowed_relations: A set of allowed relation URIs.
    :type allowed_relations: Set[str]
    :param target_total_edges: The desired total number of edges in the graph after crawling.
    :type target_total_edges: int
    :param save_callback: An optional function to call periodically to save the graph.
    :type save_callback: Optional[Callable[[nx.DiGraph], None]]
    :param save_interval_edges: Trigger save callback roughly every this many *new* edges added.
    :type save_interval_edges: int
    :param progress_callback: Callback to update progress (takes number of new edges).
    :type progress_callback: Optional[Callable[[int], None]]
    :return: The total number of *new* edges added during this crawl session.
    :rtype: int
    """
    initial_edge_count = graph.number_of_edges()
    edges_to_add = target_total_edges - initial_edge_count

    if edges_to_add <= 0:
        logging.info(f"Graph already has {initial_edge_count} edges. Target is {target_total_edges}. No crawling needed.")
        return 0

    if not allowed_relations:
        logging.error("Allowed relations set cannot be empty.")
        return 0

    # --- Setup ---
    queue = deque() # Use deque as a simple queue
    visited_nodes: Set[str] = set(graph.nodes()) # Initialize from provided graph
    processed_edges_signatures: Set[tuple] = set() # Tracks edges already processed
    # Initialize processed edges from the initial graph
    for u, v, data in graph.edges(data=True):
        relation = data.get('relation')
        if relation:
            edge_signature = tuple(sorted((u, relation, v)))
            processed_edges_signatures.add(edge_signature)

    edges_added_this_run = 0
    edges_added_since_last_save = 0

    # --- Initialize Queue ---
    if graph.number_of_nodes() > 0:
        logging.info(f"Resuming crawl. Initial graph has {graph.number_of_nodes()} nodes and {initial_edge_count} edges.")
        initial_nodes = list(graph.nodes())
        # Add nodes to queue in a deterministic order for potentially better reproducibility
        for node in sorted(initial_nodes):
            if node.startswith('/c/en/'):
                 # Add to queue only if not already visited (should be rare here, but safe)
                 if node not in visited_nodes:
                    queue.append(node)
                    visited_nodes.add(node) # Mark as visited when adding to queue initially
                 elif node not in queue: # If already visited but not in queue (e.g. start node), add it
                    queue.append(node)

        logging.info(f"Initialized queue with {len(queue)} unique English nodes from existing graph.")
        # If queue is still empty after processing existing nodes, add start node if needed
        if not queue and start_node_uri and start_node_uri.startswith('/c/en/'):
            if start_node_uri not in visited_nodes:
                queue.append(start_node_uri)
                visited_nodes.add(start_node_uri)
            elif start_node_uri not in queue: # Ensure start node is in queue if graph wasn't empty but queue became empty
                 queue.append(start_node_uri)


    else: # Starting fresh
        logging.info(f"Starting fresh crawl from {start_node_uri}")
        if not start_node_uri or not start_node_uri.startswith('/c/en/'):
             logging.error("Start node URI must be provided and be English (/c/en/) for a fresh crawl.")
             return 0
        queue.append(start_node_uri)
        visited_nodes.add(start_node_uri) # Mark start node as visited

    if not queue:
        logging.warning("Crawler queue is empty at the start. Cannot crawl.")
        return 0

    logging.info(f"Starting synchronous crawl. Target: {edges_to_add} new edges ({target_total_edges} total).")

    # --- Main Crawl Loop ---
    while queue and graph.number_of_edges() < target_total_edges:
        current_uri = queue.popleft() # Get next URI from the left

        # Skip if already visited (this check is important as nodes can be added multiple times before processing)
        # Note: We added to visited_nodes when adding to queue, so this check might seem redundant,
        # but it's safer to keep it in case of complex graph structures adding nodes back.
        # However, the primary visit check now happens *before* fetching.
        # Let's refine: We only fetch if it wasn't visited *before* being popped.
        # The visited_nodes set now tracks nodes whose *data has been fetched or attempted*.

        # Fetch data
        logging.debug(f"Processing URI: {current_uri}")
        query_limit = 50
        data = fetch_conceptnet_data_sync(current_uri + f"?limit={query_limit}")

        # Add delay between requests *after* the request attempt
        logging.debug(f"Waiting for {REQUEST_DELAY} seconds before next request...")
        time.sleep(REQUEST_DELAY)

        if data and 'edges' in data:
            edges_in_response = data['edges']
            logging.debug(f"Processing {len(edges_in_response)} edges for node {current_uri}")
            batch_added = 0

            for edge in edges_in_response:
                if graph.number_of_edges() >= target_total_edges:
                    logging.debug("Target edge count reached within edge processing loop.")
                    break # Stop processing if target reached

                relation_uri = edge.get('rel', {}).get('@id')
                start_uri = edge.get('start', {}).get('@id')
                end_uri = edge.get('end', {}).get('@id')
                weight = edge.get('weight', 1.0)
                surface_text = edge.get('surfaceText')

                # --- Filtering ---
                if not relation_uri or not start_uri or not end_uri: continue
                # Ensure both nodes are English concepts
                if not start_uri.startswith('/c/en/') or not end_uri.startswith('/c/en/'): continue
                # Ensure the edge involves the current node we are processing
                # (ConceptNet API might return edges where both start/end are different)
                # This check might be too strict if we want broader exploration, but let's keep it for now.
                # if start_uri != current_uri and end_uri != current_uri: continue
                # Ensure the relation is allowed
                if relation_uri not in allowed_relations: continue

                # --- Skip edge if surfaceText is None ---
                if surface_text is None:
                    logging.debug(f"Skipping edge from {start_uri} to {end_uri} due to None surfaceText.")
                    continue # Skip this edge

                edge_signature = tuple(sorted((start_uri, relation_uri, end_uri)))

                # --- Add to Graph ---
                if edge_signature not in processed_edges_signatures:
                    # Add nodes if they don't exist
                    start_name = start_uri.split('/')[-1]
                    end_name = end_uri.split('/')[-1]
                    if not graph.has_node(start_uri):
                        graph.add_node(start_uri, name=start_name, lang='en')
                        logging.debug(f"Added node: {start_uri}")
                    if not graph.has_node(end_uri):
                        graph.add_node(end_uri, name=end_name, lang='en')
                        logging.debug(f"Added node: {end_uri}")

                    # Add edge if it doesn't exist based on signature (start, relation, end)
                    # NetworkX default checks only (start, end), so we check relation too
                    needs_add = True
                    if graph.has_edge(start_uri, end_uri):
                        # Check if an edge with the *same relation* already exists
                        # NetworkX allows multiple edges between nodes if keys are used,
                        # but we are not using keys here, so we check the 'relation' attribute.
                        # If graph stores relation in data:
                        if graph[start_uri][end_uri].get('relation') == relation_uri:
                            needs_add = False
                            logging.debug(f"Edge {edge_signature} already exists with this relation. Skipping.")
                        # else: an edge exists, but with a different relation. Allow adding.

                    if needs_add:
                        graph.add_edge(start_uri, end_uri, relation=relation_uri, weight=weight, surfaceText=surface_text)
                        processed_edges_signatures.add(edge_signature)
                        edges_added_this_run += 1
                        edges_added_since_last_save += 1
                        batch_added += 1
                        logging.info(f"NEW Edge ({graph.number_of_edges()}/{target_total_edges}): {start_uri} -[{relation_uri}]-> {end_uri}")

                        # Call progress callback
                        if progress_callback:
                            try:
                                progress_callback(1)
                            except Exception as e:
                                logging.warning(f"Error calling progress callback: {e}")

                        # --- Add Neighbor to Queue ---
                        # Identify the neighbor node involved in the new edge
                        neighbor_uri = None
                        if start_uri == current_uri:
                            neighbor_uri = end_uri
                        elif end_uri == current_uri:
                            neighbor_uri = start_uri
                        # If the edge connects two *other* nodes (possible with ConceptNet API results),
                        # we might want to add both if they are new. Let's stick to direct neighbors for now.

                        # Add the neighbor to the queue if it's English and not visited yet
                        if neighbor_uri and neighbor_uri not in visited_nodes:
                            # Double check it's not already in the queue to avoid excessive growth
                            if neighbor_uri not in queue:
                                queue.append(neighbor_uri)
                                logging.debug(f"Added neighbor {neighbor_uri} to queue.")
                            # Mark as visited *when adding to queue* to prevent re-adding?
                            # Let's stick to marking visited *after* fetching attempt for simplicity.
                            # The check `neighbor_uri not in visited_nodes` handles this.

            # --- Periodic Save Check ---
            if batch_added > 0 and save_callback is not None:
                if edges_added_since_last_save >= save_interval_edges:
                    logging.info(f"Added >= {save_interval_edges} edges since last save ({edges_added_since_last_save} total). Triggering save callback.")
                    try:
                        save_callback(graph) # Call synchronous callback
                        edges_added_since_last_save = 0 # Reset counter after successful save
                    except Exception as e:
                        logging.error(f"Error during save callback: {e}")
                        # Don't reset counter if callback failed

        # Check if target reached after processing node's edges
        if graph.number_of_edges() >= target_total_edges:
            logging.info(f"Reached target total edges ({target_total_edges}). Stopping crawl.")
            break

    # --- Final Check ---
    if not queue and graph.number_of_edges() < target_total_edges:
        logging.warning(f"Queue became empty after processing {graph.number_of_edges()} edges, but target was {target_total_edges}.")
    elif queue and graph.number_of_edges() >= target_total_edges:
        logging.info(f"Target reached. {len(queue)} URIs remaining in queue.")

    final_edges_added = graph.number_of_edges() - initial_edge_count
    logging.info(f"Synchronous crawl finished. Added {final_edges_added} new edges.")
    return final_edges_added
