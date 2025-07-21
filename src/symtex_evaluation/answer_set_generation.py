# -*- coding: utf-8 -*-
"""
Script to process Answer Set Programming (ASP) problems, generate potential answer sets using an LLM,
and save the results. It supports multithreading and checkpointing.
"""
import concurrent.futures
import json
import logging
import os
import re # Import regex for parsing the LLM output
import time
from typing import List, Dict, Any, Set, Optional, Literal  # Added Optional for potentially missing fields

from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser
from tqdm import tqdm
from jinja2 import Environment, BaseLoader
from pydantic import BaseModel, Field, create_model  # Added Field for better validation/defaults
from apiHelper.langchain.custom_llm import CustomLLM
# Import the specific function from your utils
from src.utils.json_utils import read_jsonl, JsonUtilsError, _JSON_LIB # Import necessary components
# Import the prompts dictionary
from src.llm_prompt.prompts_answer_set_generation import prompts_answer_set_generation
from src.utils.langchain_utils import handle_cb

# --- Logging Setup ---
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True) # Ensure log directory exists
log_file_path = os.path.join(log_directory, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'), # Ensure UTF-8 encoding for file handler
        logging.StreamHandler() # Keep console output
    ]
)
logger = logging.getLogger(__name__) # Get logger for this module

# --- Pydantic Model for Result ---
class AnswerSet(BaseModel):
    """Represents a single answer set, which is a collection of atoms.

    Attributes:
        atoms: A list of strings, where each string is an atom in the answer set.
    """
    atoms: list[str]


# --- Main Framework Class ---
class AnswerSetGenerationFramework:
    """
    框架类，用于处理 ASP 问题的答案集生成任务。
    加载数据，调用大模型进行推理，并处理结果。
    """
    def __init__(self, model_name: str, json_parsing_model_name: str, alignment_batch_size: int = 5, llm_options_limit: int = 30):
        """
        初始化框架。

        :param model_name: 用于实例化主 CustomLLM 的模型名称。
        :type model_name: str
        :param json_parsing_model_name: 用于 JSON 解析的 CustomLLM 的模型名称。
        :type json_parsing_model_name: str
        :param alignment_batch_size: 对齐时处理预测原子的批次大小。
        :type alignment_batch_size: int
        :param llm_options_limit: 提供给LLM进行对齐的最大候选原子数（包括None）。
        :type llm_options_limit: int
        """
        self.logger = logging.getLogger(__name__)
        self.alignment_batch_size = alignment_batch_size
        self.llm_options_limit = max(1, llm_options_limit) # Ensure at least 1 option (e.g., None)
        self.logger.info(f"Alignment batch size set to: {self.alignment_batch_size}")
        self.logger.info(f"LLM alignment options limit set to: {self.llm_options_limit}")

        try:
            self.llm = CustomLLM(model_name)
            self.json_parser = JsonOutputParser()
            self.json_parsing_model_name = json_parsing_model_name
            self.json_llm = CustomLLM(json_parsing_model_name)
            self.json_llm.response_format = AnswerSet
            self.chain_to_json_answer_set = self.json_llm | self.json_parser
            self.logger.info(f"CustomLLM initialized with model: {model_name}")
            self.logger.info(f"JSON parsing LLM initialized with model: {json_parsing_model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing CustomLLM with model {model_name} or JSON parser model {json_parsing_model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize CustomLLM: {e}") from e

        # Pre-compile regex for parsing answer set: { lit1, lit2, ... }
        # Handles optional whitespace around literals and commas
        self.answer_set_regex = re.compile(r"{\s*([^}]+?)\s*}")
        self.literal_split_regex = re.compile(r"\s*,\s*") # Splits by comma with optional surrounding whitespace

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从 JSONL 文件加载数据，使用 src.utils.json_utils.read_jsonl。

        :param file_path: JSONL 文件的路径。
        :type file_path: str
        :return: 包含数据项字典的列表。
        :rtype: List[Dict[str, Any]]
        """
        data = []
        try:
            data = read_jsonl(file_path)
            self.logger.info(f"Successfully loaded {len(data)} items from {file_path} using json_utils.")
        except FileNotFoundError:
            self.logger.error(f"Error: Input file not found at {file_path}")
            raise
        except (_JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e:
             self.logger.error(f"Error decoding JSON in {file_path}: {e}")
             raise
        except JsonUtilsError as e:
            self.logger.error(f"An error occurred reading {file_path} using json_utils: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred while loading data from {file_path}: {e}")
            raise
        return data


    def run(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个数据项，执行答案集生成和可选的对齐。

        :param data_item: 包含问题和潜在答案集的数据字典。
        :return: 包含原始数据、LLM 响应、预测答案集和评估结果（如果适用）的字典。
        """
        item_id = data_item.get('id', 'N/A')
        facts = data_item.get('facts', [])
        rules = data_item.get('rules', [])

        llm_input_and_response = []

        # --- 1. Prepare Prompt ---
        template_string = prompts_answer_set_generation.get('Generation', {}).get('prompt')
        if not template_string:
            raise ValueError("Prompt template 'Generation' not found in prompts_answer_set_generation.")

        jinja_env = Environment(loader=BaseLoader())
        template = jinja_env.from_string(template_string)
        llm_input_prompt = template.render(
            facts=facts,
            rules=rules
        )

        # --- 2. Call LLM ---
        with get_openai_callback() as cb_raw:
            llm_raw_response = self.llm.invoke(llm_input_prompt)

        llm_input_and_response.append({
            'input': llm_input_prompt,
            'response': llm_raw_response,
            **handle_cb(cb_raw)
        })

        prompt_text_to_json = (
            # f"{llm_input_prompt}\\n\\n"
            f"Reasoning:\\n{llm_raw_response}\\n\\n"
            f"Based on the reasoning, determine the final state of the decision.\\n"
            f"Json format: \\n{AnswerSet.model_json_schema()}\\n\\n"
            f"Final Json Answer:\\n"
        )

        with get_openai_callback() as cb_to_json:
            answerset_json: dict = self.chain_to_json_answer_set.invoke(prompt_text_to_json)

        llm_input_and_response.append({
            'input': prompt_text_to_json,
            'response': json.dumps(answerset_json),
            **handle_cb(cb_to_json)
        })

        # --- 3. Parse LLM Response ---
        # Remove quotes from each atom in the predicted answer set
        predicted_answer_set = [atom.replace('"', '') for atom in answerset_json.get('atoms', [])]

        results = {}
        if 'answer_sets' in data_item:
            # --- Build Normalized Ground Truth Map ---
            normalized_to_original_gt_map: Dict[str, str] = {}
            full_ground_truth_atoms_set_contains_none = False
            for answer_set_atoms_collection in data_item['answer_sets']:
                for atom in answer_set_atoms_collection:
                    processed_atom = atom.replace('"', '') # Remove quotes
                    if processed_atom is None:
                        full_ground_truth_atoms_set_contains_none = True
                        continue # Skip normalization for None
                    if not isinstance(processed_atom, str):
                         # Log or handle non-string atoms if necessary
                         self.logger.warning(f"Item {item_id}: Encountered non-string ground truth atom: {processed_atom}. Skipping.")
                         continue

                    # Normalize by removing all standard spaces
                    normalized_atom = processed_atom.replace(" ", "") # Use simple replace

                    # Store mapping from normalized to original. Handle potential conflicts if needed.
                    if normalized_atom in normalized_to_original_gt_map and normalized_to_original_gt_map[normalized_atom] != processed_atom:
                        self.logger.warning(f"Item {item_id}: Multiple ground truth atoms normalize to '{normalized_atom}'. "
                                             f"Overwriting '{normalized_to_original_gt_map[normalized_atom]}' with '{processed_atom}'.")
                    normalized_to_original_gt_map[normalized_atom] = processed_atom

            # --- Global Direct Matching First ---
            all_answerset_aligned = {} # Stores final alignments (direct + LLM)
            atoms_needing_llm = []     # List of predicted atoms (original objects) needing LLM
            predicted_answer_set_list = list(predicted_answer_set)

            # self.logger.info(f"Item {item_id}: Starting global direct matching for {len(predicted_answer_set_list)} predicted atoms...")
            direct_match_count = 0
            for pred_atom_obj in predicted_answer_set_list:
                pred_atom_str = str(pred_atom_obj)

                # Handle None case
                if pred_atom_obj is None:
                    if full_ground_truth_atoms_set_contains_none:
                        all_answerset_aligned[pred_atom_str] = None # Direct match for None
                        direct_match_count += 1
                    else:
                        atoms_needing_llm.append(pred_atom_obj) # Needs LLM
                    continue

                # Normalize and attempt direct match
                normalized_pred_atom = pred_atom_str.replace(" ", "")
                if normalized_pred_atom in normalized_to_original_gt_map:
                    original_gt_match = normalized_to_original_gt_map[normalized_pred_atom]
                    all_answerset_aligned[pred_atom_str] = original_gt_match
                    direct_match_count += 1
                else:
                    atoms_needing_llm.append(pred_atom_obj) # Needs LLM
            
            # self.logger.info(f"Item {item_id}: Global direct matching complete. {direct_match_count} atoms directly matched. {len(atoms_needing_llm)} atoms require LLM alignment.")

            # --- LLM Alignment for Remaining Atoms (if any) ---
            if atoms_needing_llm:
                # --- Outer Loop: Batch processing for PREDICTED atoms needing LLM ---
                outer_batch_size = self.alignment_batch_size
                num_outer_batches = (len(atoms_needing_llm) + outer_batch_size - 1) // outer_batch_size
                for i in range(0, len(atoms_needing_llm), outer_batch_size):
                    current_llm_batch_pred_atoms = atoms_needing_llm[i : i + outer_batch_size]
                    # Note: current_llm_batch_pred_atoms now contains the actual atom objects
                    
                    outer_batch_num = i // outer_batch_size + 1
                    # self.logger.info(f"Item {item_id}: Processing LLM Outer Batch {outer_batch_num}/{num_outer_batches} ({len(current_llm_batch_pred_atoms)} atoms).")

                    # Initialize pending map for this LLM outer batch
                    pending_llm_alignment_map_current_outer_batch = {str(atom): atom for atom in current_llm_batch_pred_atoms}
                    
                    # --- Inner Loop: Batch processing for CANDIDATE (Ground Truth) keys for LLM ---
                    # (Prepare candidate_key_batches_for_llm_original - logic remains same)
                    all_original_gt_atoms_list_no_none = list(normalized_to_original_gt_map.values()) # Get original strings
                    gt_atoms_per_llm_call = max(1, self.llm_options_limit - 1)
                    candidate_key_batches_for_llm_original = []
                    for gt_idx in range(0, len(all_original_gt_atoms_list_no_none), gt_atoms_per_llm_call):
                        candidate_key_batches_for_llm_original.append(all_original_gt_atoms_list_no_none[gt_idx : gt_idx + gt_atoms_per_llm_call])
                    if not candidate_key_batches_for_llm_original and all_original_gt_atoms_list_no_none:
                        # self.logger.warning(f"Item {item_id}, LLM Outer Batch {outer_batch_num}: No candidate key batches generated though GT atoms exist. Check logic.")
                        candidate_key_batches_for_llm_original.append([])
                    if not all_original_gt_atoms_list_no_none and not candidate_key_batches_for_llm_original:
                        candidate_key_batches_for_llm_original.append([])
                    num_gt_batches = len(candidate_key_batches_for_llm_original)
                    
                    # (Inner loop iterations remain the same, operating on pending_llm_alignment_map_current_outer_batch)
                    for gt_batch_iter_idx, current_original_gt_atom_batch in enumerate(candidate_key_batches_for_llm_original):
                        if not pending_llm_alignment_map_current_outer_batch:
                            # self.logger.info(f"Item {item_id}, LLM Outer Batch {outer_batch_num}: All pending atoms successfully aligned. Halting GT batch iteration.")
                            break # Exit inner loop (GT batches)

                        gt_batch_num_display = gt_batch_iter_idx + 1
                        atoms_for_this_llm_step = list(pending_llm_alignment_map_current_outer_batch.values())
                        current_llm_gt_options = list(set(current_original_gt_atom_batch) | {None if full_ground_truth_atoms_set_contains_none else None})
                        if None not in current_llm_gt_options and full_ground_truth_atoms_set_contains_none:
                            current_llm_gt_options.append(None)
                        elif None in current_llm_gt_options and not full_ground_truth_atoms_set_contains_none:
                            current_llm_gt_options = [opt for opt in current_llm_gt_options if opt is not None]
                        
                        # self.logger.info(f"Item {item_id}, LLM Outer Batch {outer_batch_num}, GT Batch {gt_batch_num_display}/{num_gt_batches}: "
                        #                  f"Aligning {len(atoms_for_this_llm_step)} pred atoms against {len(current_llm_gt_options)} GT options.")

                        field_definitions_for_llm_step = {}
                        try:
                            for pred_atom_obj_for_llm in atoms_for_this_llm_step:
                                literal_options = tuple(opt for opt in current_llm_gt_options if opt is not None or full_ground_truth_atoms_set_contains_none)
                                if not literal_options:
                                    # self.logger.warning(f"Item {item_id}, LLM Outer Batch {outer_batch_num}, GT Batch {gt_batch_num_display}: No valid Literal options for atom '{str(pred_atom_obj_for_llm)}'. Skipping its definition for this LLM step.")
                                    continue
                                field_definitions_for_llm_step[str(pred_atom_obj_for_llm)] = (
                                    Literal[literal_options],
                                    Field(..., description=f"Aligned result of '{str(pred_atom_obj_for_llm)}' using GT batch {gt_batch_num_display}")
                                )
                        except Exception as e:
                            # self.logger.error(f"Item {item_id}, LLM Outer Batch {outer_batch_num}, GT Batch {gt_batch_num_display}: Error preparing Literal for dynamic model: {e}", exc_info=True)
                            continue

                        if not field_definitions_for_llm_step:
                            # self.logger.warning(f"Item {item_id}, LLM Outer Batch {outer_batch_num}, GT Batch {gt_batch_num_display}: No fields defined for dynamic model. Skipping LLM call.")
                            continue
                        
                        # (LLM call and result processing logic remains the same)
                        AlignmentModelForLLMStep = create_model(
                            f"Align_OB{outer_batch_num}_GB{gt_batch_num_display}",
                            **field_definitions_for_llm_step,
                            __doc__=f"Alignment for item {item_id}, outer pred batch {outer_batch_num}, GT batch {gt_batch_num_display}"
                        )
                        chain_alignment_llm_step = CustomLLM(self.json_parsing_model_name)
                        chain_alignment_llm_step.response_format = AlignmentModelForLLMStep
                        chain_alignment_llm_step = chain_alignment_llm_step | self.json_parser
                        alignment_prompt_llm_step = (
                            f"--- Task ---\n"
                            f"Align the *current set of predicted atoms* with the *current batch of possible ground truth atoms*.\n\n"
                            f"Predicted Atoms To Align Now:\n{ [str(a) for a in atoms_for_this_llm_step] }\n\n"
                            f"Possible Ground Truth Atoms (Current Batch, including None):\n{current_llm_gt_options}\n\n"
                            f"--- Instructions ---\n"
                            f"For each 'Predicted Atom', find the best matching 'Possible Ground Truth Atom' from the current batch. "
                            f"Ensure the match is exact or semantically identical. If there is ANY ambiguity or the match is not perfect, default to `null`.\n"
                            f"If no ground truth atom in this batch corresponds, use `null` (JSON equivalent of None).\n"
                            f"Respond ONLY with the JSON object requested.\n\n"
                            f"--- Final JSON Alignment (Current Step) ---\n"
                        )
                        try:
                            with get_openai_callback() as cb_align_step:
                                llm_results_this_step: Dict[str, Any] = chain_alignment_llm_step.invoke(alignment_prompt_llm_step)
                            llm_input_and_response.append({
                                'input': alignment_prompt_llm_step,
                                'response': json.dumps(llm_results_this_step),
                                **handle_cb(cb_align_step),
                                'context': f"Item {item_id}, OuterBatch {outer_batch_num}, GTBatch {gt_batch_num_display}"
                            })
                            newly_resolved_in_this_step_count = 0
                            for pred_atom_str_key, aligned_value in llm_results_this_step.items():
                                if aligned_value is not None:
                                    all_answerset_aligned[pred_atom_str_key] = aligned_value
                                    if pred_atom_str_key in pending_llm_alignment_map_current_outer_batch:
                                        del pending_llm_alignment_map_current_outer_batch[pred_atom_str_key]
                                        newly_resolved_in_this_step_count +=1
                                    # self.logger.info(f"Item {item_id}, LLM Outer Batch {outer_batch_num}, GT Batch {gt_batch_num_display}: Atom '{pred_atom_str_key}' aligned to '{aligned_value}'.")
                            # self.logger.info(f"Item {item_id}, LLM Outer Batch {outer_batch_num}, GT Batch {gt_batch_num_display}: {newly_resolved_in_this_step_count} atoms got non-None alignment in this step.")
                        except Exception as align_exc_step:
                            # self.logger.error(f"Item {item_id}, LLM Outer Batch {outer_batch_num}, GT Batch {gt_batch_num_display}: Error during LLM alignment step: {align_exc_step}", exc_info=True)
                            llm_input_and_response.append({
                                'input': alignment_prompt_llm_step,
                                'response': f"Error during alignment step: {align_exc_step}",
                                'error_details': str(align_exc_step),
                                'context': f"Item {item_id}, OuterBatch {outer_batch_num}, GTBatch {gt_batch_num_display}"
                            })
                    # --- End of Inner Loop (GT Batches) ---
                    
                    # After trying all GT batches for this outer batch, finalize remaining as None
                    for final_pending_pred_atom_str in list(pending_llm_alignment_map_current_outer_batch.keys()):
                        all_answerset_aligned[final_pending_pred_atom_str] = None # Final result is None
                        # self.logger.info(f"Item {item_id}, LLM Outer Batch {outer_batch_num}: Atom '{final_pending_pred_atom_str}' remains unaligned after all GT batches. Finalizing as None.")
                    pending_llm_alignment_map_current_outer_batch.clear()
                # --- End of Outer Loop (Predicted Atom Batches needing LLM) ---
            # else: # No atoms needed LLM alignment
                # self.logger.info(f"Item {item_id}: No atoms required LLM alignment after direct matching.")
                
            # --- Consolidate final results --- 
            # all_answerset_aligned now contains results for all original predicted atoms (either from direct match or LLM path)
            aligned_answer_set = [all_answerset_aligned.get(str(p_atom)) for p_atom in predicted_answer_set_list]
            # Optional: Add a final check for completeness, though .get() handles missing keys gracefully
            if len(aligned_answer_set) != len(predicted_answer_set_list):
                self.logger.error(f"Item {item_id}: Final aligned_answer_set length ({len(aligned_answer_set)}) does not match original predicted length ({len(predicted_answer_set_list)}). This is unexpected.")
            
            results['aligned_answer_set'] = aligned_answer_set
            results['golden_answet_sets'] = data_item['answer_sets']

        results = {
            'id': data_item['id'],
            'llm_input_and_response': llm_input_and_response,
            'predicted_answer_set': predicted_answer_set,
            'asp_program': '\n'.join(facts+rules),
            **results
        }

        return results

    def process_dataset(self, input_file_path: str, output_file_path: str, num_threads: int = 4, save_interval: int = 100):
        """
        加载数据集，使用多线程处理，并支持断点续存和增量保存。

        :param input_file_path: 输入 JSONL 文件的路径。
        :type input_file_path: str
        :param output_file_path: 输出 JSONL 文件的路径。
        :type output_file_path: str
        :param num_threads: 用于处理数据集的线程数。
        :type num_threads: int
        :param save_interval: 每处理多少个项目就强制刷新缓冲区到文件。
        :type save_interval: int
        :return: None. 结果直接写入文件。
        :rtype: None
        """
        self.logger.info(f"Starting dataset processing for Answer Set Generation.")
        self.logger.info(f"Input file: {input_file_path}")
        self.logger.info(f"Output file: {output_file_path}")
        self.logger.info(f"Threads: {num_threads}, Save Interval: {save_interval}")

        # --- 1. 加载输入数据 ---
        self.logger.info(f"Attempting to load input data from: {input_file_path}")
        try:
            loaded_data = self.load_data(input_file_path)
        except Exception as e:
            self.logger.critical(f"Failed to load input data from {input_file_path}: {e}. Aborting processing.")
            return

        if not loaded_data:
            self.logger.warning("Input data file loaded successfully but is empty. No items to process.")
            return

        total_items_input = len(loaded_data)
        self.logger.info(f"Successfully loaded {total_items_input} items from input file.")

        # --- 2. 加载已处理的 ID (如果输出文件存在) ---
        processed_ids: Set[str] = set()
        if os.path.exists(output_file_path):
            self.logger.info(f"Output file {output_file_path} exists. Loading processed IDs for checkpointing...")
            try:
                existing_data = read_jsonl(output_file_path)
                processed_ids = {item['id'] for item in existing_data if isinstance(item, dict) and item.get('id') is not None}
                self.logger.info(f"Loaded {len(processed_ids)} unique processed IDs from existing output file.")
            except FileNotFoundError:
                 self.logger.warning(f"Output file {output_file_path} disappeared unexpectedly during ID loading. Starting fresh.")
            except (JsonUtilsError, _JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e:
                 self.logger.error(f"Error reading/decoding JSON in existing output file {output_file_path}. "
                                   f"Consider backing up and deleting/fixing the file before restarting. Error: {e}", exc_info=True)
                 return
            except Exception as e:
                 self.logger.exception(f"An unexpected error occurred while loading processed IDs from {output_file_path}: {e}")
                 return
        else:
            self.logger.info(f"Output file {output_file_path} does not exist. Starting a new processing run.")

        # --- 3. 过滤掉已处理项 ---
        items_to_process = [item for item in loaded_data if isinstance(item, dict) and item.get('id') not in processed_ids]
        total_items_to_process = len(items_to_process)

        if not items_to_process:
            self.logger.info(f"All {total_items_input} items from input file are already present (based on ID) in the output file {output_file_path}. No new items require processing.")
            return

        self.logger.info(f"Identified {total_items_to_process} new items to process (out of {total_items_input} total).")

        processed_count = 0
        successful_saves = 0
        failed_count = 0
        start_time = time.time()

        # --- 4. 确保输出目录存在 ---
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
             try:
                 os.makedirs(output_dir, exist_ok=True)
                 self.logger.info(f"Ensured output directory exists: {output_dir}")
             except OSError as e:
                 self.logger.error(f"Could not create output directory {output_dir}: {e}. Aborting processing.")
                 return

        # --- 5. 使用多线程处理并写入文件 ---
        try:
            with open(output_file_path, 'a', encoding='utf-8', buffering=1) as outfile, \
                 concurrent.futures.ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix='ASPGenWorker') as executor:

                self.logger.info(f"Submitting {total_items_to_process} items to the thread pool...")
                future_to_item_id = {executor.submit(self.run, data_item): data_item.get('id', f'unidentified_index_{i}')
                                     for i, data_item in enumerate(items_to_process)}
                self.logger.info("All items submitted.")

                self.logger.info("Waiting for results...")
                for future in tqdm(concurrent.futures.as_completed(future_to_item_id), total=total_items_to_process, desc="Generating Answer Sets", unit="item"):
                    item_id = future_to_item_id[future]
                    processed_count += 1

                    try:
                        result: dict = future.result() # result is the dict returned by run()

                        if result and isinstance(result, dict): # Check if result is a non-empty dict
                            # Serialize the result dictionary to a JSON string
                            json_line = json.dumps(result, ensure_ascii=False)
                            outfile.write(json_line + '\n')
                            successful_saves += 1

                            # Log errors captured within the result object
                            if result.get('error'):
                                self.logger.warning(f"Item {item_id}: Processing completed with error: {result['error']}")
                                # Count as failed if an error was logged during run()
                                failed_count += 1


                            if successful_saves % save_interval == 0:
                                outfile.flush()
                                self.logger.info(f"Flushed output buffer. Saved: {successful_saves}/{total_items_to_process} ({processed_count} futures completed).")
                        else:
                            # This case might occur if run() returns None or unexpected type, though currently it shouldn't
                            self.logger.error(f"Item {item_id}: Received unexpected result type from run(): {type(result)}. Skipping save.")
                            failed_count += 1


                    except Exception as exc:
                        # Catches exceptions raised by future.result() itself (errors in run not caught there)
                        failed_count += 1
                        self.logger.error(f'Item {item_id} generated an exception during future processing: {exc}', exc_info=True)

                self.logger.info("Processing loop finished. Performing final flush.")
                outfile.flush()

        except IOError as e:
            self.logger.error(f"Fatal I/O error interacting with output file {output_file_path}: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during the dataset processing loop: {e}", exc_info=True)

        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info("--- Processing Summary ---")
            self.logger.info(f"Total items in input: {total_items_input}")
            self.logger.info(f"Items already processed (resumed): {len(processed_ids)}")
            self.logger.info(f"Items submitted for processing in this run: {total_items_to_process}")
            self.logger.info(f"Futures completed: {processed_count}")
            # Note: Successful saves might include items where run() logged an error but still returned a result structure
            self.logger.info(f"Items written to output file: {successful_saves}")
            self.logger.info(f"Items failed (exceptions or run errors): {failed_count}")
            self.logger.info(f"Total processing time: {duration:.2f} seconds.")

# --- Main Execution ---
# Example Usage (Consider adding argparse for flexibility)
if __name__ == "__main__":
    logger.info("Starting Answer Set Generation script.")

    # --- Configuration ---
    # TODO: Replace with argparse or environment variables
    INPUT_FILE = "path/to/your/input_data.jsonl"  # <-- CHANGE THIS
    OUTPUT_FILE = "path/to/your/output_results.jsonl" # <-- CHANGE THIS
    MODEL_NAME = "local_qwen2_5_7b" # Or your desired LLM model
    JSON_PARSING_MODEL_NAME = "local_qwen2_5_7b" # Or your desired JSON parsing model
    NUM_THREADS = 4 # Adjust based on your system resources
    SAVE_INTERVAL = 50 # Adjust as needed

    # --- File Existence Check ---
    if not os.path.exists(INPUT_FILE):
        logger.critical(f"Input file not found: {INPUT_FILE}. Please check the path.")
        # Exit if input file doesn't exist to prevent running with incorrect config
        exit(1)
    else:
         logger.info(f"Found input file: {INPUT_FILE}")

    # --- Framework Initialization and Execution ---
    try:
        framework = AnswerSetGenerationFramework(
            model_name=MODEL_NAME, 
            json_parsing_model_name=JSON_PARSING_MODEL_NAME,
            alignment_batch_size=5,  # Default: 5, or pass a new value
            llm_options_limit=30    # Default: 30, or pass a new value
        )
        framework.process_dataset(
            input_file_path=INPUT_FILE,
            output_file_path=OUTPUT_FILE,
            num_threads=NUM_THREADS,
            save_interval=SAVE_INTERVAL
        )
        logger.info("Script execution finished.")
    except RuntimeError as e:
        logger.critical(f"Framework initialization failed: {e}. Aborting.")
    except Exception as e:
        logger.critical(f"An critical error occurred during script execution: {e}", exc_info=True) 