# import json # No longer needed for loading
import json # Need standard json for dump/dumps if not using orjson for writing
# Add necessary imports for multithreading, logging, os, typing, tqdm
import concurrent.futures
import logging
import os
import time # Import time for potential delays or timing
from typing import List, Dict, Any, Literal, Set, Optional # Updated import

from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm # Import tqdm for progress bar
from jinja2 import Environment, BaseLoader # Import Jinja2
from pydantic import BaseModel # Added import
from apiHelper.langchain.custom_llm import CustomLLM
from langchain_core.output_parsers import JsonOutputParser
# Import the specific function from your utils
from src.utils.json_utils import read_jsonl, JsonUtilsError, _JSON_LIB # Import necessary components (Removed write_jsonl as it's not used directly here)

# Import the prompts dictionary
from src.llm_prompt.prompts_answer_set_selection import prompts_answer_set_decision
from src.utils.langchain_utils import handle_cb

# Setup basic logging
# Configure logging to output to both console and a file
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True) # Ensure log directory exists
log_file_path = os.path.join(log_directory, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Keep console output
    ]
)
logger = logging.getLogger(__name__) # Get logger for this module

class DecisionResultState(BaseModel):
    """
    Represents the decision state for a query, indicating whether the answer is "Yes" or "No".
    """
    decision: Literal["Yes", "No"]

class AnswerSetDecisionFramework:
    """
    用于处理事实状态查询任务的框架类。
    加载数据，调用大模型进行推理，并处理结果。
    """
    def __init__(self, model_name: str, json_parsing_model_name: str):
        """
        初始化框架。

        :param model_name: 用于实例化主 CustomLLM 的模型名称。
        :type model_name: str
        :param json_parsing_model_name: 用于 JSON 解析的 CustomLLM 的模型名称。
        :type json_parsing_model_name: str
        """
        # Use the module-level logger
        self.logger = logging.getLogger(__name__)
        try:
            self.llm = CustomLLM(model_name)

            self.json_parser = JsonOutputParser()
            # 使用参数化的模型名称
            self.json_llm = CustomLLM(json_parsing_model_name)
            self.json_llm.response_format = DecisionResultState
            self.chain_to_json_query_result_state = self.json_llm | self.json_parser
            self.logger.info(f"CustomLLM initialized with main model: {model_name}")
            self.logger.info(f"JSON parsing LLM initialized with model: {json_parsing_model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing CustomLLM with model {model_name} or JSON parser model {json_parsing_model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize CustomLLM: {e}") from e

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
            # Use the imported read_jsonl function
            data = read_jsonl(file_path)
            self.logger.info(f"Successfully loaded {len(data)} items from {file_path} using json_utils.")
        except FileNotFoundError:
            self.logger.error(f"Error: Input file not found at {file_path}")
            # Reraise or return empty list? Reraising might be better for process_dataset
            raise
        except (_JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e: # Catch potential decode errors from orjson/json
             self.logger.error(f"Error decoding JSON in {file_path}: {e}")
             raise # Reraise to indicate failure
        except JsonUtilsError as e:
            # Catch custom errors from json_utils
            self.logger.error(f"An error occurred reading {file_path} using json_utils: {e}")
            raise # Reraise
        except Exception as e:
            # Catch any other unexpected errors
            self.logger.exception(f"An unexpected error occurred while loading data from {file_path}: {e}")
            raise # Reraise
        return data

    def run(self, data_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理单个数据项，执行决策任务。

        :param data_item: 单个数据项的字典。
        :type data_item: Dict[str, Any]
        :return: 包含处理结果和元数据（id, target_query, label, prediction 等）的字典，如果出错则返回 None。
        :rtype: Optional[Dict[str, Any]]
        """
        item_id = data_item.get('id', 'N/A')
        target_query = data_item.get('answer_set_decision', {}).get('answerset') # 获取目标查询
        label = data_item.get('answer_set_decision', {}).get('type') # 获取标签

        try:
            facts = data_item.get('facts', [])
            rules = data_item.get('rules', [])

            template_string = prompts_answer_set_decision.get('CoT', {}).get('prompt')

            jinja_env = Environment(loader=BaseLoader())
            template = jinja_env.from_string(template_string)
            rendered_prompt = template.render(
                facts=facts,
                rules=rules,
                candidate_answer_set=target_query
            )

            final_prompt = f"{rendered_prompt}"

            # LLM Call 1: Get initial reasoning
            self.logger.debug(f"Item {item_id}: Invoking main LLM ({self.llm.model_name}) for decision.")
            with get_openai_callback() as cb_raw_text:
                result_text: str = self.llm.invoke(final_prompt)

            self.logger.debug(f"Item {item_id}: Raw response from main LLM: {result_text}")

            # LLM Call 2: Format output as JSON
            self.logger.debug(f"Item {item_id}: Invoking JSON parsing LLM ({self.json_llm.model_name}).")
            prompt_text_to_json = (
                # f"{final_prompt}\\n\\n"
                f"Reasoning:\\n{result_text}\\n\\n"
                f"Based on the reasoning, determine the final state of the decision.\\n"
                f"Json format: \\n{DecisionResultState.model_json_schema()}\\n\\n"
                f"Final Json Answer:\\n"
            )

            # Invoke the chain specifically for JSON parsing
            with get_openai_callback() as cb_to_json:
                result_json = self.chain_to_json_query_result_state.invoke(prompt_text_to_json)
            decision_output = result_json['decision'] # Extract the decision

            # Map decision to 'Correct' or 'Error'
            if decision_output == "Yes":
                prediction = "Correct"
            else: # Includes "No" and any other unexpected values
                prediction = "Error"
                if decision_output != "No": # Log if the output wasn't the expected "No"
                    self.logger.warning(f"Item {item_id}: Received unexpected decision value '{decision_output}'. Mapping to 'Error'.")

            # Construct the full result dictionary including metadata
            return {
                'id': item_id,
                'target_query': target_query,
                'label': label,
                'llm_input_and_response': [
                    {'input': final_prompt, 'response': result_text, **handle_cb(cb_raw_text)},
                    {'input': prompt_text_to_json,
                     'response': json.dumps(result_json.model_dump() if isinstance(result_json, DecisionResultState) else str(result_json)),
                     **handle_cb(cb_to_json)
                     } # Safely dump or stringify
                ],
                'asp_program': '\n'.join(facts + rules), # Join facts and rules for context
                'prediction': prediction,
            }

        except KeyError as e:
            self.logger.error(f"Error processing item {item_id}: Missing key {e}", exc_info=True)
            return None
        except Exception as e:
            # Catch any other unexpected error during the processing of a single item
            self.logger.exception(f"An unexpected error occurred during processing item {item_id}: {e}")
            return None # Return None to indicate failure for this item

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
        # --- 1. 加载输入数据 ---
        self.logger.info(f"Attempting to load input data from: {input_file_path}")
        try:
            loaded_data = self.load_data(input_file_path)
        except Exception as e:
            self.logger.critical(f"Failed to load input data from {input_file_path}: {e}. Aborting processing.")
            return # Stop if loading fails

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
                # Use read_jsonl which should handle internal errors more gracefully
                existing_data = read_jsonl(output_file_path)
                # Ensure item is a dict and 'id' exists and is not None before adding
                processed_ids = {item['id'] for item in existing_data if isinstance(item, dict) and item.get('id') is not None}
                self.logger.info(f"Loaded {len(processed_ids)} unique processed IDs from existing output file.")
            except FileNotFoundError:
                 # This case should theoretically not be hit due to os.path.exists, but handle defensively
                 self.logger.warning(f"Output file {output_file_path} disappeared unexpectedly during ID loading. Starting fresh.")
            except (JsonUtilsError, _JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e:
                 self.logger.error(f"Error reading/decoding JSON in existing output file {output_file_path}. "
                                   f"Consider backing up and deleting/fixing the file before restarting. Error: {e}", exc_info=True)
                 # Abort if the checkpoint file is corrupt, as continuing could lead to data loss or duplication
                 return
            except Exception as e:
                 # Catch any other unexpected errors during ID loading
                 self.logger.exception(f"An unexpected error occurred while loading processed IDs from {output_file_path}: {e}")
                 # Abort as the state is uncertain
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
        self.logger.info(f"Starting dataset processing with {num_threads} threads.")
        self.logger.info(f"Results will be appended to: {output_file_path}")
        self.logger.info(f"Output buffer will be flushed every {save_interval} successful saves.")


        processed_count = 0 # Counter for items submitted to thread pool
        successful_saves = 0 # Counter for items successfully processed and saved
        failed_count = 0 # Counter for items that failed processing in run() or during future.result()
        start_time = time.time()

        # --- 4. 确保输出目录存在 ---
        output_dir = os.path.dirname(output_file_path)
        if output_dir: # Ensure directory path is not empty
             try:
                 os.makedirs(output_dir, exist_ok=True)
                 self.logger.info(f"Ensured output directory exists: {output_dir}")
             except OSError as e:
                 self.logger.error(f"Could not create output directory {output_dir}: {e}. Aborting processing.")
                 return

        # --- 5. 使用多线程处理并写入文件 ---
        try:
            # Open file in append mode ('a') with UTF-8 encoding and line buffering (often default, but can be explicit)
            with open(output_file_path, 'a', encoding='utf-8', buffering=1) as outfile, \
                 concurrent.futures.ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix='FactQueryWorker') as executor:

                self.logger.info(f"Submitting {total_items_to_process} items to the thread pool...")
                # Submit jobs only for items that need processing
                # Map future to the original data_item's ID for error tracking
                future_to_item_id = {executor.submit(self.run, data_item): data_item.get('id', f'unidentified_index_{i}')
                                     for i, data_item in enumerate(items_to_process)}
                self.logger.info("All items submitted.")

                # Process completed futures as they finish, using tqdm for progress
                self.logger.info("Waiting for results...")
                for future in tqdm(concurrent.futures.as_completed(future_to_item_id), total=total_items_to_process, desc="Processing items", unit="item"):
                    item_id = future_to_item_id[future] # Get the original ID associated with the future
                    processed_count += 1 # Increment processed count as each future completes

                    try:
                        result: dict | None = future.result() # result is the dict returned by run() or None

                        if result is not None:
                            # Ensure the result is a dictionary before proceeding
                            if not isinstance(result, dict):
                                self.logger.error(f"Item {item_id}: Expected dict from run(), but got {type(result)}. Skipping save.")
                                failed_count += 1
                                continue

                            # Serialize the result dictionary to a JSON string
                            # Use standard json.dumps for better compatibility and handling non-ASCII
                            json_line = json.dumps(result, ensure_ascii=False)
                            # Write the JSON string as a line to the output file
                            outfile.write(json_line + '\n')
                            # No need to manually flush if line buffering (buffering=1) is effective,
                            # but periodic explicit flush provides stronger guarantee, especially for larger save_intervals.
                            successful_saves += 1

                            # Flush buffer periodically based on successful saves
                            if successful_saves % save_interval == 0:
                                outfile.flush()
                                self.logger.info(f"Flushed output buffer. Saved: {successful_saves}/{total_items_to_process} ({processed_count} futures completed).")

                        else:
                            # Log items that returned None (indicating an error during run)
                            self.logger.warning(f'Item {item_id} processing failed within run() method. Not saved.')
                            failed_count += 1

                    except Exception as exc:
                        # This catches exceptions *raised* by the future.result() call itself
                        # (i.e., exceptions within the self.run method that were not caught there, or other thread-related issues)
                        failed_count += 1
                        self.logger.error(f'Item {item_id} generated an exception during future processing: {exc}', exc_info=True)
                        # Do not write anything for items that caused exceptions here

                # Final flush after the loop finishes to catch any remaining buffered lines
                self.logger.info("Processing loop finished. Performing final flush.")
                outfile.flush()

        except IOError as e:
            self.logger.error(f"Fatal I/O error interacting with output file {output_file_path}: {e}", exc_info=True)
            # Consider how to handle partial results - data might be lost
        except Exception as e:
            # Catch any other unexpected errors during the main processing block
            self.logger.error(f"An unexpected error occurred during the dataset processing loop: {e}", exc_info=True)

        finally:
            # This block executes whether an exception occurred or not
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info("--- Processing Summary ---")
            self.logger.info(f"Total items in input: {total_items_input}")
            self.logger.info(f"Items already processed (resumed): {len(processed_ids)}")
            self.logger.info(f"Items submitted for processing in this run: {total_items_to_process}")
            self.logger.info(f"Futures completed: {processed_count}")
            self.logger.info(f"Successfully processed and saved: {successful_saves}")
            self.logger.info(f"Failed items: {failed_count}")
            self.logger.info(f"Total processing time: {duration:.2f} seconds.")
            # No return value needed as results are written directly
