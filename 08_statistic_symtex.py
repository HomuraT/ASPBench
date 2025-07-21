# -*- coding: utf-8 -*-
"""
Script to load and count records from JSONL files in specified task directories.

This script reads all .jsonl files recursively from three predefined directories:
- datasets/a_symtex_task_answerset_generation
- datasets/a_symtex_task_answerset_selection
- datasets/a_symtex_task_fact_state_querying
It merges the content of files within each directory into lists in memory.
The script stores these lists in a dictionary accessible after execution (e.g.,
in a debugger) and prints the total record count for each task to the console.
It utilizes parallel processing for reading JSONL files for efficiency.
Generates statistical summaries, plots, and a final Markdown report. # Added Markdown report
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for type checking

# --- Constants ---
BASE_INPUT_DIR: Path = Path("datasets") / "SymTex"
BASE_OUTPUT_DIR: Path = Path("experiments") / "symtex_statistic"

TASK_DIRS: Dict[str, str] = {
    "answerset_generation": "answerset_generation_symbolic.jsonl",
    "answerset_selection": "answerset_selection_symbolic.jsonl",
    "fact_state_querying": "fact_state_querying_symbolic.jsonl",
}

# --- Setup Output Directory and Logging ---
# Create a timestamped directory for this run
TIMESTAMP: str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
RUN_OUTPUT_DIR: Path = BASE_OUTPUT_DIR / TIMESTAMP
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging to file and console
LOG_FILE_PATH: Path = RUN_OUTPUT_DIR / "statistics_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler() # Also log to console
    ]
)

# Attempt to import the required functions from the utils module
try:
    # Assuming the script is run from the project root directory (SymTex)
    from src.utils.json_utils import read_jsonl_parallel, write_jsonl, JsonUtilsError
except ImportError as e:
    logging.error(f"Error importing from src.utils.json_utils: {e}")
    logging.error("Please ensure the script is run from the project root directory 'SymTex' "
                  "and the 'src' directory is accessible.")
    exit(1) # Exit if imports fail

# --- End Setup ---


# --- Plotting Function ---
def plot_statistic(data: pd.Series, variable_name: str, task_output_dir: Path, run_output_dir: Path, plot_type: str = 'auto') -> Union[Path, None]:
    """
    Generates and saves a plot for a given statistical variable.

    Args:
        data (pd.Series): The data series to plot.
        variable_name (str): The name of the variable (used for title and filename).
        task_output_dir (Path): The directory to save the plot image for the specific task.
        run_output_dir (Path): The base output directory for the current run (used for relative path calculation).
        plot_type (str): The type of plot ('histogram', 'boxplot', 'barplot', 'auto').
                         'auto' tries to determine the best plot based on data type.

    Returns:
        Union[Path, None]: The relative path to the saved plot file (relative to run_output_dir)
                           if successful, otherwise None.
    """
    if data.empty:
        logging.warning(f"Skipping plot for '{variable_name}' as data is empty.")
        return None

    plt.figure(figsize=(10, 6))
    # Ensure task_output_dir exists (should be created in main, but double-check)
    task_output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename_abs = task_output_dir / f"{variable_name}.png"

    # Clean data: remove NaN/inf for plotting if they exist
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data_clean.empty:
        logging.warning(f"Skipping plot for '{variable_name}' as data is empty after cleaning NaN/inf.")
        plt.close() # Close the figure if not used
        return None

    try:
        if plot_type == 'auto':
            if pd.api.types.is_numeric_dtype(data_clean) and data_clean.nunique() > 10: # Heuristic for continuous data
                plot_type = 'histogram'
            elif pd.api.types.is_numeric_dtype(data_clean): # Numeric but few unique values, treat as categorical/discrete
                 plot_type = 'barplot' # Or maybe boxplot if distribution matters more? Let's start with barplot.
            else: # Categorical or object type
                plot_type = 'barplot'

        logging.info(f"Plotting '{variable_name}' as {plot_type}...")

        if plot_type == 'histogram':
            # Use seaborn for better aesthetics and automatic binning
            sns.histplot(data_clean, kde=True)
            plt.title(f'Distribution of {variable_name}')
            plt.xlabel(variable_name)
            plt.ylabel('Frequency')
        elif plot_type == 'boxplot':
             # Boxplot is good for showing distribution summaries
            sns.boxplot(y=data_clean)
            plt.title(f'Box Plot of {variable_name}')
            plt.ylabel(variable_name)
        elif plot_type == 'barplot':
             # Suitable for categorical data or discrete numerical data
            # If numeric, might need binning first, but let's try value_counts directly
            if pd.api.types.is_numeric_dtype(data_clean):
                # For discrete numeric, show counts per value
                counts = data_clean.value_counts().sort_index()
                sns.barplot(x=counts.index, y=counts.values)
                plt.title(f'Counts per Value for {variable_name}')
                plt.xlabel(variable_name)
                plt.ylabel('Count')
                # Rotate x-labels if many categories
                if len(counts) > 15:
                    plt.xticks(rotation=45, ha='right')
            else:
                # For categorical data
                counts = data_clean.value_counts()
                sns.barplot(x=counts.index, y=counts.values)
                plt.title(f'Counts per Category for {variable_name}')
                plt.xlabel(variable_name)
                plt.ylabel('Count')
                # Rotate x-labels if many categories or long labels
                if len(counts) > 10 or max(len(str(x)) for x in counts.index) > 10:
                     plt.xticks(rotation=45, ha='right')
        else:
            logging.warning(f"Unsupported plot type '{plot_type}' for variable '{variable_name}'. Skipping plot.")
            plt.close()
            return None

        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.savefig(plot_filename_abs)
        logging.info(f"Saved plot to {plot_filename_abs}")
        plt.close() # Close the figure to free memory

        # Calculate relative path
        try:
            relative_plot_path = plot_filename_abs.relative_to(run_output_dir)
            return relative_plot_path
        except ValueError:
            logging.error(f"Could not determine relative path for {plot_filename_abs} from {run_output_dir}")
            return None # Or return absolute path if preferred fallback

    except Exception as e:
        logging.error(f"Failed to generate plot for '{variable_name}': {e}", exc_info=True)
        plt.close() # Ensure figure is closed even on error
        return None

# --- End Plotting Function ---


# --- Statistics Processing Function ---
def process_task_statistics(task_key: str, data: List[Dict[str, Any]], task_output_dir: Path, run_output_dir: Path) -> Tuple[Dict[str, Dict[str, float]], List[Path]]:
    """
    Calculates statistics, logs them, generates plots, and returns summary data and plot paths.

    Args:
        task_key (str): The identifier for the task (e.g., 'fact_state_querying').
        data (List[Dict[str, Any]]): The list of records loaded for this task.
        task_output_dir (Path): The directory to save plots for this task.
        run_output_dir (Path): The base output directory for the current run.

    Returns:
        Tuple[Dict[str, Dict[str, float]], List[Path]]: A tuple containing:
            - A dictionary of summary statistics (min, max, mean) for each variable.
            - A list of relative paths (Path objects) to the generated plot files.
    """
    logging.info(f"Processing statistics for task: {task_key}")
    df = pd.DataFrame(data)

    summary_stats: Dict[str, Dict[str, float]] = {}
    plot_files: List[Path] = []

    stats_to_calculate: List[str] = []
    derived_stats: Dict[str, Any] = {} # For stats like ratios or flattened lists

    # Define variables based on task
    if task_key == "fact_state_querying":
        stats_to_calculate = [
            'num_facts', 'num_rules', 'num_related_predicates',
            'max_depth_of_rule_graph', 'average_depth_of_rule_graph',
            'label', 'max_ary_for_predicates', 'num_min_facts_for_query',
            'num_rules_with_default_negation',  # Needed for ratio
            'source_type',  # Added new field
            'is_disjunctive_stratified',  # Added new field
            'is_positive',  # Added new field
            'is_tight',  # Added new field
            'is_stratified'  # Added new field
        ]
    elif task_key in ["answerset_generation", "answerset_selection"]:
        stats_to_calculate = [
            'num_facts', 'num_rules', 'num_rules_with_default_negation',  # Needed for ratio
            'max_ary_for_predicates', 'num_answer_sets',
            'answer_set_fact_counts',  # Special handling needed
            'source_type',  # Added new field
            'is_disjunctive_stratified',  # Added new field
            'is_positive',  # Added new field
            'is_tight',  # Added new field
            'is_stratified'  # Added new field
        ]
    else:
        logging.warning(f"Unknown task key '{task_key}' for statistics processing. Skipping.")
        return {}, [] # Return empty results

    extracted_data: Dict[str, List[Any]] = {stat: [] for stat in stats_to_calculate}
    all_answer_set_counts: List[int] = [] # Specific for answer set tasks

    # --- Data Extraction Loop ---
    for record in data:
        for stat in stats_to_calculate:
            value = record.get(stat)
            if value is None:
                # Log missing value, decide how to handle (skip, use NaN, etc.)
                # For simplicity, we might skip or pandas will handle it as NaN later
                logging.debug(f"Missing value for '{stat}' in record: {record.get('id', 'N/A')}")
                extracted_data[stat].append(np.nan) # Append NaN for missing values
                continue

            # Special handling
            if stat == 'num_min_facts_for_query' and value == 0:
                 extracted_data[stat].append(np.nan) # Treat 0 as missing for this stat's analysis
            elif stat == 'answer_set_fact_counts':
                if isinstance(value, list):
                    extracted_data[stat].append(value) # Keep the list for now, maybe plot len?
                    all_answer_set_counts.extend(value) # Flatten for overall distribution
                else:
                    extracted_data[stat].append(np.nan) # Invalid format
                    logging.warning(f"Invalid format for 'answer_set_fact_counts' (expected list): {value}")
            else:
                extracted_data[stat].append(value)
    # --- End Data Extraction Loop ---

    # Convert extracted data to DataFrame for easier processing
    stats_df = pd.DataFrame(extracted_data)

    # --- Calculate Derived Statistics ---
    if 'num_rules' in stats_df.columns and 'num_rules_with_default_negation' in stats_df.columns:
        # Calculate ratio, handle division by zero
        # Ensure denominators are not zero before division
        valid_rules_mask = stats_df['num_rules'] != 0
        stats_df['rule_default_negation_ratio'] = np.nan # Initialize with NaN
        stats_df.loc[valid_rules_mask, 'rule_default_negation_ratio'] = (
            stats_df.loc[valid_rules_mask, 'num_rules_with_default_negation'] / stats_df.loc[valid_rules_mask, 'num_rules']
        )
        # Add to the list of stats to process further
        stats_to_calculate.append('rule_default_negation_ratio')

    # Special handling for answer_set_fact_counts - maybe plot the distribution of counts within lists?
    # For now, let's create a separate series for the flattened list
    if all_answer_set_counts:
         derived_stats['all_answer_set_fact_counts'] = pd.Series(all_answer_set_counts)

    # --- Calculate and Log Summary Statistics ---
    logging.info(f"\n--- Summary Statistics for Task: {task_key} ---")
    
    # Define categorical fields for special handling
    categorical_fields = ['source_type', 'is_disjunctive_stratified', 'is_positive', 'is_tight', 'is_stratified', 'label']

    for col in stats_df.columns:
        if col not in stats_to_calculate: # Skip columns not requested or intermediate
             continue

        series = stats_df[col].dropna() # Drop NaN for calculations

        # Special case: 'answer_set_fact_counts' is a list of lists, handle differently
        if col == 'answer_set_fact_counts':
             # Calculate stats on the *length* of the lists (number of answer sets with counts)
             series_lengths = stats_df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
             series_lengths = series_lengths.replace(0, np.nan).dropna() # Ignore records with 0 length lists or NaNs
             stat_name_len = f"{col}_list_length"
             if not series_lengths.empty:
                 min_val, max_val, mean_val = series_lengths.min(), series_lengths.max(), series_lengths.mean()
                 summary_stats[stat_name_len] = {'min': min_val, 'max': max_val, 'mean': mean_val}
                 logging.info(f"  {stat_name_len}: Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}")
                 # Plot the distribution of list lengths
                 plot_path = plot_statistic(series_lengths, stat_name_len, task_output_dir, run_output_dir)
                 if plot_path:
                     plot_files.append(plot_path)
             continue # Skip standard processing for the list itself

        plot_path: Union[Path, None] = None # Initialize plot_path
        
        # Special handling for categorical fields
        if col in categorical_fields:
            # Calculate value counts and percentages
            value_counts = series.value_counts()
            total = len(series)
            percentages = (value_counts / total * 100).round(2)
            
            # Log detailed statistics
            logging.info(f"\n  {col} Statistics:")
            logging.info("  Value Counts and Percentages:")
            for value, count in value_counts.items():
                percentage = percentages[value]
                logging.info(f"    {value}: {count} ({percentage}%)")
            
            # Store counts in summary stats
            summary_stats[col] = {
                'total': total,
                'unique_values': len(value_counts),
                'value_counts': value_counts.to_dict()
            }
            
            # Generate bar plot for categorical data
            plot_path = plot_statistic(series, col, task_output_dir, run_output_dir, plot_type='barplot')
            
        elif pd.api.types.is_numeric_dtype(series) and not series.empty:
            min_val, max_val, mean_val = series.min(), series.max(), series.mean()
            summary_stats[col] = {'min': min_val, 'max': max_val, 'mean': mean_val}
            logging.info(f"  {col}: Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}")
            # Plot numeric data
            plot_path = plot_statistic(series, col, task_output_dir, run_output_dir)
        elif not series.empty: # Handle other categorical data (like 'label')
            # Log value counts for categorical data
            counts = series.value_counts()
            logging.info(f"  {col}: Value Counts:\n{counts.to_string()}")
            # Plot categorical data
            plot_path = plot_statistic(series, col, task_output_dir, run_output_dir, plot_type='barplot')
        else:
            logging.info(f"  {col}: No valid data points found.")

        if plot_path:
            plot_files.append(plot_path)

    # Process derived stats (like the flattened answer set counts)
    for stat_name, derived_series in derived_stats.items():
         if not derived_series.empty:
             min_val, max_val, mean_val = derived_series.min(), derived_series.max(), derived_series.mean()
             summary_stats[stat_name] = {'min': min_val, 'max': max_val, 'mean': mean_val}
             logging.info(f"  {stat_name}: Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}")
             plot_path = plot_statistic(derived_series, stat_name, task_output_dir, run_output_dir)
             if plot_path:
                 plot_files.append(plot_path)
         else:
             logging.info(f"  {stat_name}: No valid data points found.")

    logging.info(f"--- End Summary Statistics for Task: {task_key} ---")
    return summary_stats, plot_files

# --- End Statistics Processing Function ---


def load_records_from_directory(input_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Reads a JSONL file and returns its contents as a list of records.

    Args:
        input_dir (Union[str, Path]): The path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: A list containing all valid JSON records found in the file.
                              Returns an empty list if the file doesn't exist or is invalid.

    Raises:
        FileNotFoundError: If the input file does not exist.
        JsonUtilsError: If errors occur during JSONL reading.
        Exception: For other unexpected errors during file processing.
    """
    input_path = Path(input_dir)

    if not input_path.is_file():
        logging.warning(f"Input file not found: {input_path}. Returning empty list.")
        return []

    logging.info(f"Reading file: {input_path}")

    try:
        # Use parallel reading for potentially large files
        data: List[Dict[str, Any]] = read_jsonl_parallel(str(input_path))
        logging.info(f"Successfully read {len(data)} records from {input_path}")
        return data
    except JsonUtilsError as e:
        logging.error(f"Error reading JSONL file {input_path}: {e}")
        return []
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {input_path}: {e}")
        return []


# --- Markdown Report Generation Function ---
def generate_markdown_report(
    output_dir: Path,
    all_stats: Dict[str, Dict[str, Dict[str, float]]],
    all_plots: Dict[str, List[Path]]
) -> None:
    """
    Generates a Markdown report summarizing statistics and embedding plots.

    Args:
        output_dir (Path): The main output directory for the run where the report will be saved.
        all_stats (Dict[str, Dict[str, Dict[str, float]]]): A dictionary where keys are task names,
            and values are dictionaries of statistics (stat_name -> {'min': float, 'max': float, 'mean': float}).
        all_plots (Dict[str, List[Path]]): A dictionary where keys are task names,
            and values are lists of relative Path objects to the plot images.

    Returns:
        None
    """
    report_path = output_dir / "statistics_report.md"
    logging.info(f"Generating Markdown report at: {report_path}")
    md_content = []

    # --- Report Header ---
    md_content.append(f"# SymTex Dataset Statistics Report")
    md_content.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append(f"Output Directory: `{output_dir}`")
    md_content.append("\n---\n")

    # --- Loop Through Tasks ---
    for task_key in TASK_DIRS.keys(): # Iterate in defined order
        md_content.append(f"## Task: {task_key}\n")

        # --- Statistics Table ---
        task_stats = all_stats.get(task_key)
        if task_stats:
            md_content.append("### Summary Statistics\n")
            
            # First, handle numeric statistics
            md_content.append("#### Numeric Statistics\n")
            md_content.append("| Statistic Name | Min | Max | Mean |")
            md_content.append("|---|---|---|---|")
            
            for stat_name, values in sorted(task_stats.items()):
                # Skip categorical fields for now
                if isinstance(values, dict) and 'value_counts' in values:
                    continue
                    
                # Check against broader numeric types including numpy types
                min_val_raw = values.get('min')
                max_val_raw = values.get('max')
                mean_val_raw = values.get('mean')

                min_val = f"{min_val_raw:.2f}" if isinstance(min_val_raw, (int, float, np.number)) else 'N/A'
                max_val = f"{max_val_raw:.2f}" if isinstance(max_val_raw, (int, float, np.number)) else 'N/A'
                mean_val = f"{mean_val_raw:.2f}" if isinstance(mean_val_raw, (int, float, np.number)) else 'N/A'

                md_content.append(f"| `{stat_name}` | {min_val} | {max_val} | {mean_val} |")
            md_content.append("\n") # Add space after table

            # Then, handle categorical statistics
            md_content.append("#### Categorical Statistics\n")
            for stat_name, values in sorted(task_stats.items()):
                if isinstance(values, dict) and 'value_counts' in values:
                    md_content.append(f"\n##### {stat_name}\n")
                    md_content.append("| Value | Count | Percentage |")
                    md_content.append("|---|---|---|")
                    
                    total = values.get('total', 0)
                    for value, count in values['value_counts'].items():
                        percentage = (count / total * 100) if total > 0 else 0
                        md_content.append(f"| {value} | {count} | {percentage:.2f}% |")
                    md_content.append("\n")
        else:
            md_content.append("*No statistics calculated for this task.*\n")

        # --- Plots ---
        task_plots = all_plots.get(task_key)
        if task_plots:
            md_content.append("### Plots\n")
            for plot_path_relative in sorted(task_plots, key=lambda p: p.name):
                # Convert Path object to string, use forward slashes for Markdown compatibility
                plot_path_str = plot_path_relative.as_posix()
                plot_name = plot_path_relative.stem # Get filename without extension
                md_content.append(f"#### {plot_name}\n")
                # Use HTML img tag with width attribute for resizing
                md_content.append(f'<img src="{plot_path_str}" alt="{plot_name}" width="50%">\n')
            md_content.append("\n") # Add space after plots section
        else:
            md_content.append("*No plots generated for this task.*\n")

        md_content.append("\n---\n") # Separator between tasks

    # --- Write to File ---
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_content))
        logging.info(f"Successfully wrote Markdown report to {report_path}")
    except IOError as e:
        logging.error(f"Failed to write Markdown report to {report_path}: {e}")

# --- End Markdown Report Generation Function ---


def main() -> None:
    """
    Main function to orchestrate the data loading, statistics calculation,
    plotting, and Markdown report generation for all defined tasks.
    """
    logging.info(f"Starting SymTex data statistics generation. Output dir: {RUN_OUTPUT_DIR}")
    logging.info(f"Log file: {LOG_FILE_PATH}")

    # Dictionaries to hold data, counts, stats, and plots, local to main
    task_data: Dict[str, List[Dict[str, Any]]] = {}
    task_counts: Dict[str, int] = {}
    all_tasks_summary_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_tasks_plot_files: Dict[str, List[Path]] = {}

    total_tasks = len(TASK_DIRS)
    tasks_processed = 0
    tasks_with_errors = 0

    for task_key, filename in TASK_DIRS.items():
        input_file = BASE_INPUT_DIR / filename
        task_output_dir = RUN_OUTPUT_DIR / task_key # Specific output dir for this task's plots
        task_output_dir.mkdir(exist_ok=True) # Ensure task-specific output dir exists

        logging.info(f"\n--- Processing Task: {task_key} ---")
        logging.info(f"Input file: {input_file}")
        logging.info(f"Output directory for plots: {task_output_dir}")

        try:
            # Load the actual data
            loaded_data: List[Dict[str, Any]] = load_records_from_directory(input_file)
            task_data[task_key] = loaded_data
            record_count = len(loaded_data)
            task_counts[task_key] = record_count
            tasks_processed += 1 # Increment even if 0 records, as loading was attempted
            logging.info(f"Loaded {record_count} records for task '{task_key}'.")

            # --- Process Statistics and Plotting ---
            if record_count > 0:
                logging.info(f"Starting statistics calculation and plotting for task '{task_key}'...")
                # Pass RUN_OUTPUT_DIR to process_task_statistics
                summary_stats, plot_files = process_task_statistics(
                    task_key, loaded_data, task_output_dir, RUN_OUTPUT_DIR
                )
                all_tasks_summary_stats[task_key] = summary_stats
                all_tasks_plot_files[task_key] = plot_files
                logging.info(f"Finished statistics and plotting for task '{task_key}'.")
            else:
                logging.warning(f"Skipping statistics for task '{task_key}' as no records were loaded.")
                # Store empty results for consistency
                all_tasks_summary_stats[task_key] = {}
                all_tasks_plot_files[task_key] = []
            # --- End Statistics Processing ---

            logging.info(f"--- Task '{task_key}' processed successfully. ---")

        except Exception as e:
            logging.error(f"An unexpected error occurred during task '{task_key}': {e}", exc_info=True)
            task_data[task_key] = []
            task_counts[task_key] = -1 # Indicate error
            all_tasks_summary_stats[task_key] = {} # Store empty results on error
            all_tasks_plot_files[task_key] = []
            tasks_with_errors += 1
            logging.info(f"--- Task '{task_key}' failed during processing ---")

    # --- Generate Final Markdown Report ---
    logging.info("\n--- Generating Final Markdown Report ---")
    generate_markdown_report(RUN_OUTPUT_DIR, all_tasks_summary_stats, all_tasks_plot_files)
    logging.info("--- Finished Markdown Report Generation ---")

    # --- Final Summary Logging ---
    logging.info("\n=================== Summary ===================")
    logging.info("Record Counts per Task:")
    for task_key, count in task_counts.items():
        if count >= 0:
            logging.info(f"- {task_key}: {count} records loaded")
        else:
            logging.info(f"- {task_key}: Error during loading/processing")
    logging.info("---------------------------------------------")
    logging.info(f"Total tasks attempted: {total_tasks}")
    logging.info(f"Tasks processed (attempted loading): {tasks_processed}")
    logging.info(f"Tasks with errors during processing: {tasks_with_errors}")
    logging.info("=============================================")
    logging.info(f"Statistics generation finished. Check the log file: {LOG_FILE_PATH}")
    logging.info(f"Plots saved in subdirectories under: {RUN_OUTPUT_DIR}")
    logging.info(f"Markdown report saved as: {RUN_OUTPUT_DIR / 'statistics_report.md'}")


if __name__ == "__main__":
    main()
