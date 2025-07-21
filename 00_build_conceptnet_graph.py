# Main script to build the ConceptNet graph using synchronous crawler
import logging
import os
# import asyncio # No longer needed
import networkx as nx
from tqdm import tqdm # Import tqdm for progress bar
from collections import Counter # Import Counter for counting
# Import the synchronous crawler function
from src.conceptnet_utils.crawler import crawl_conceptnet_sync
from src.conceptnet_utils.storage import save_graph_to_graphml, load_graph_from_graphml

# --- Configuration ---
START_NODE_URI = "/c/en/bird"
# Relations relevant for ASP, including negations
ALLOWED_RELATIONS = {
    "/r/IsA",
    "/r/CapableOf",
    "/r/HasA",
    "/r/PartOf",
    "/r/UsedFor",
    "/r/Antonym",
    "/r/RelatedTo",
    "/r/NotHasProperty",
    "/r/NotCapableOf",
    "/r/NotDesires",
    "/r/NotUsedFor",
    "/r/NotIsA",
    # Add more if needed, e.g., /r/MannerOf, /r/LocatedNear, /r/MadeOf
}
MAX_EDGES = 30000  # Target edge count (adjust as needed)
OUTPUT_DIR = "datasets/conceptnet"
OUTPUT_FILENAME = "bird_graph.graphml"
OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
CRAWL_BATCH_SIZE = 50  # Number of edges to attempt crawling per iteration
SAVE_INTERVAL_EDGES = 100 # Save the graph every time this many new edges are added

# --- Logging Setup ---
# Explicitly set the root logger level to DEBUG to ensure debug messages are captured
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO) # Basic config first (can be INFO)
logger = logging.getLogger() # Get the root logger
logger.setLevel(logging.INFO) # Explicitly set level to DEBUG
logging.info("Logging level set to DEBUG.") # Confirm level setting

# --- Sync Save Callback ---
# Keep track of the edge count at the last successful save
last_saved_edge_count_sync = -1

def save_graph_sync(graph_to_save: nx.DiGraph):
    """
    Synchronous callback function to save the graph.

    :param graph_to_save: The graph object to save.
    :type graph_to_save: nx.DiGraph
    """
    global last_saved_edge_count_sync
    current_edge_count = graph_to_save.number_of_edges()
    # Simple check: save if edge count is different from last save
    # The crawler itself controls the interval, this callback just performs the save.
    if current_edge_count != last_saved_edge_count_sync:
        logging.info(f"Callback triggered: Saving intermediate graph with {current_edge_count} edges...")
        try:
            save_graph_to_graphml(graph_to_save, OUTPUT_FILEPATH)
            logging.info(f"Intermediate graph saved successfully to {OUTPUT_FILEPATH}")
            last_saved_edge_count_sync = current_edge_count # Update count after successful save
        except Exception as e:
            logging.error(f"Failed to save intermediate graph via callback: {e}")
    else:
         logging.debug(f"Callback triggered but edge count ({current_edge_count}) hasn't changed since last save ({last_saved_edge_count_sync}). Skipping save.")


# --- Main Function ---
def main(): # No longer async
    """Main function to orchestrate graph building."""
    global last_saved_edge_count_sync # Use sync counter
    logging.info("Starting Synchronous ConceptNet graph building process...")
    logging.info(f"Target total edges: {MAX_EDGES}")
    logging.info(f"Output file: {OUTPUT_FILEPATH}")

    # 1. Load existing graph or initialize a new one
    graph = load_graph_from_graphml(OUTPUT_FILEPATH)
    initial_nodes = 0
    initial_edges = 0
    if graph is not None:
        initial_nodes = graph.number_of_nodes()
        initial_edges = graph.number_of_edges()
        logging.info(f"Loaded existing graph with {initial_nodes} nodes and {initial_edges} edges.")
        # Initialize the sync save counter based on the loaded graph
        last_saved_edge_count_sync = initial_edges
    else:
        logging.info("No existing graph found. Starting fresh.")
        graph = nx.DiGraph()
        last_saved_edge_count_sync = -1 # Reset for fresh graph

    # 2. Run the asynchronous crawler with progress bar
    if graph.number_of_edges() < MAX_EDGES:
        logging.info(f"Current edges: {graph.number_of_edges()}. Target: {MAX_EDGES}. Starting crawler...")

        # Initialize tqdm progress bar
        with tqdm(total=MAX_EDGES, initial=graph.number_of_edges(), unit='edge', desc="Crawling ConceptNet") as pbar:
            # Define the progress callback function
            def update_pbar(n: int):
                pbar.update(n)

            # Call the synchronous crawler function
            edges_added = crawl_conceptnet_sync(
                graph=graph,
                start_node_uri=START_NODE_URI,
                allowed_relations=ALLOWED_RELATIONS,
                target_total_edges=MAX_EDGES,
                save_callback=save_graph_sync, # Pass the sync save callback
                save_interval_edges=SAVE_INTERVAL_EDGES,
                progress_callback=update_pbar
            )
            # Ensure pbar reaches 100% if crawling finished successfully
            pbar.n = graph.number_of_edges() # Set final value accurately
            pbar.refresh() # Refresh display

        logging.info(f"Crawler finished. Added {edges_added} new edges.")
    else:
        logging.info("Graph already meets or exceeds target edge count. Skipping crawl.")
        edges_added = 0

    # 3. Final Save (ensure the very latest state is saved)
    final_edge_count = graph.number_of_edges()
    logging.info(f"Final graph state: {graph.number_of_nodes()} nodes, {final_edge_count} edges.")

    # Save if edges were added OR if the current count differs from the last *sync* save count
    if edges_added > 0 or final_edge_count != last_saved_edge_count_sync:
        logging.info(f"Performing final save of the graph to {OUTPUT_FILEPATH}...")
        try:
            # Final save call
            save_graph_to_graphml(graph, OUTPUT_FILEPATH)
            logging.info("Final graph saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save the final graph: {e}")
    else:
        logging.info("No new edges added or graph unchanged since last save via callback. Skipping final save.")

    # 4. Count specified relations
    logging.info("Counting specified relations in the final graph...")
    relations_to_count = ALLOWED_RELATIONS
    relation_counts = Counter()
    for u, v, data in graph.edges(data=True):
        # Corrected key from 'rel' to 'relation' based on debug output
        relation = data.get('relation')
        if relation in relations_to_count:
            relation_counts[relation] += 1

    logging.info("--- Relation Counts ---")
    # Iterate through the requested relations to ensure all are printed, even if count is 0
    for relation in sorted(list(relations_to_count)): # Sort for consistent output order
        count = relation_counts.get(relation, 0) # Get count, default to 0 if not found
        logging.info(f"{relation}: {count}")
    logging.info("-----------------------")


    logging.info("Synchronous graph building process completed.")


# --- Main Execution ---
if __name__ == "__main__":
    main() # Directly call the main function
