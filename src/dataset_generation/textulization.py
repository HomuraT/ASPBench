# import json # No longer needed for loading
import json # Need standard json for dump/dumps if not using orjson for writing
import copy # Import the copy module for deepcopy
import itertools # Import itertools for flattening lists
import re # Import regular expressions
from typing import List, Dict, Any, Literal, Set, Union # Updated import
from jinja2 import Environment, BaseLoader # Import Jinja2
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, create_model, Field # Added Field import
from apiHelper.langchain.custom_llm import CustomLLM
# Import the specific function from your utils
from src.utils.json_utils import read_jsonl, JsonUtilsError, _JSON_LIB, write_jsonl # Import necessary components
import os # Import os for path operations
# import time # No longer needed for delay
import math # Import math for ceiling division
import logging # Import logging
import sys # Import sys for stdout stream handler
import concurrent.futures # Import for multithreading
from tqdm import tqdm # Import tqdm for progress bar

# Import the prompts dictionary
from src.llm_prompt.prompts_textulization import prompts_textulization


# Helper function for key normalization
def _normalize_symbolic_key(symbolic_item: str) -> str:
    """
    规范化符号条目字符串以用作一致的键。
    移除引号，移除可选的结尾句点，并去除首尾空格。

    :param symbolic_item: 原始符号条目字符串。
    :type symbolic_item: str
    :return: 规范化后的键字符串。
    :rtype: str
    """
    # 移除引号
    key = symbolic_item.replace('"', '')
    # 移除结尾句点（如果存在）
    if key.endswith('.'):
        key = key[:-1]
    # 去除首尾空格
    key = key.strip()
    return key

# --- Helper function for item transformation ---
def _transform_item_for_llm(item_string: str) -> str:
    """
    Transforms an ASP rule or fact string into a more explicit format for the LLM,
    marking negations clearly and adding a detailed translation hint.

    Rule Example: "-p(X) :- q(X), not -r(X)." becomes
    "-p(X) :- q(X), not -r(X). :: [if] q(X), <default negation> <strong negation> r(X) [then] <strong negation> p(X) :: \"If q(X) is true and there is no evidence that r(X) is explicitly false, then p(X) is explicitly false.\""

    Fact Example: "-fact(a)." becomes
    "fact: -fact(a). :: <strong negation> fact(a) :: \"fact(a) is explicitly false.\""

    Positive Fact Example: "bird(tweety)." becomes
    "fact: bird(tweety). :: bird(tweety) :: \"bird(tweety) is true.\""

    If the input is not a rule and not a strongly negated fact, it returns the original string.

    :param item_string: The original symbolic item string.
    :type item_string: str
    :return: The transformed string with original, processed parts, and translation hint.
    :rtype: str
    """
    item_string_stripped = item_string.strip()

    def process_negations(text: str) -> str:
        """Helper to process negations within a rule part."""
        # Order matters: handle 'not -' first
        processed = text.replace("not -", "<default negation> <strong negation> ")
        processed = processed.replace("not ", "<default negation> ")
        processed = processed.replace("-", "<strong negation> ")
        # Clean up potential double spaces introduced by replacements
        processed = ' '.join(processed.split())
        return processed

    def _translate_processed_part(part_string: str) -> str:
        """Translates a processed rule part (body or head) into natural language."""
        literals = []
        # Handle potential disjunction in the head first
        if ' | ' in part_string:
            literals = [l.strip() for l in part_string.split(' | ')]
            joiner = " or "
        else:
            # Split body by comma, handling potential spaces and ensuring not to split inside parentheses
            # Use regex: split on comma followed by optional whitespace, negative lookahead for closing parenthesis without opening one
            literals = [l.strip() for l in re.split(r",\s*(?![^()]*\))", part_string)]
            joiner = " and "

        translated_literals = []
        for literal in literals:
            if not literal: continue # Skip empty strings if split results in them

            # Handle comparisons first as they don't use the standard markers
            comparison_ops = [' > ', ' < ', ' = ', ' != ', ' >= ', ' <= ']
            op_found = None
            for op in comparison_ops:
                # Ensure the operator is surrounded by spaces or at boundaries
                # to avoid matching within predicate names or arguments
                if f' {op.strip()} ' in f' {literal} ':
                     op_found = op
                     break
                # Check boundaries if needed, though less likely with spaces
                # elif literal.startswith(f"{op.strip()} ") or literal.endswith(f" {op.strip()}"):
                #    op_found = op
                #    break


            if op_found:
                # Basic direct translation for comparisons
                parts = literal.split(op_found, 1)
                op_text = op_found.strip()
                # Use standard comparison symbols in translation
                translated_literals.append(f"{parts[0].strip()} {op_text} {parts[1].strip()}")
                continue # Move to next literal after handling comparison

            # Revised logic: Translate each literal completely, then join.
            translation_phrase = ""
            if literal.startswith("<default negation> <strong negation>"):
                # Use slicing based on known marker length for precision
                marker_len = len("<default negation> <strong negation> ")
                base = literal[marker_len:].strip()
                translation_phrase = f"there is no evidence that {base} is explicitly false"
            elif literal.startswith("<strong negation>"):
                marker_len = len("<strong negation> ")
                base = literal[marker_len:].strip()
                translation_phrase = f"{base} is explicitly false"
            elif literal.startswith("<default negation>"):
                marker_len = len("<default negation> ")
                base = literal[marker_len:].strip()
                translation_phrase = f"there is no evidence that {base} is true"
            elif literal == "<false>":
                 translation_phrase = "<false>" # Keep special marker
            else:
                # Positive literal
                base = literal.strip()
                translation_phrase = f"{base} is true"

            # Append the fully constructed phrase for the current literal
            translated_literals.append(translation_phrase)

        # Join the correctly translated independent phrases
        final_translated = joiner.join([t for t in translated_literals if t != "<false>"])
        # Capitalize the first letter if it's part of a sentence structure (like If...)
        # This is handled when constructing the final hint string.
        return final_translated


    # Check if it's an integrity constraint (starts with :-)
    if item_string_stripped.startswith(":-"):
        try:
            # Extract body part (everything after :-)
            body_part = item_string_stripped[2:].strip()
            # Remove trailing period if present
            if body_part.endswith('.'):
                body_part = body_part[:-1].strip()

            processed_body = process_negations(body_part)

            # Ensure original rule ends with a period
            original_rule_formatted = item_string_stripped if item_string_stripped.endswith('.') else item_string_stripped + '.'

            # Generate detailed translation hint
            translated_body_hint = _translate_processed_part(processed_body)
            # Capitalize first letter
            translation_hint = f'"It cannot be the case that {translated_body_hint}."'

            # Format for integrity constraint
            return f"rule: {original_rule_formatted} :: [if] {processed_body} [then] <false> :: {translation_hint}"
        except Exception as e:
            # Log error (using logger if available, or print)
            # print(f"Error transforming integrity constraint '{item_string}': {e}")
            return item_string # Fallback

    # Check if it's a standard rule (contains :- but doesn't start with it)
    elif ":-" in item_string_stripped:
        try:
            head_part, body_part = item_string_stripped.split(":-", 1)
            head_part = head_part.strip()
            body_part = body_part.strip()
            # Remove trailing period from body if present
            if body_part.endswith('.'):
                body_part = body_part[:-1].strip()

            processed_head = process_negations(head_part)
            processed_body = process_negations(body_part) # Comparison operators pass through unchanged

            # Ensure original rule ends with a period
            original_rule_formatted = item_string_stripped if item_string_stripped.endswith('.') else item_string_stripped + '.'

            # Generate detailed translation hint
            translated_body_hint = _translate_processed_part(processed_body)
            translated_head_hint = _translate_processed_part(processed_head)
            # Capitalize first letter of the condition
            # Ensure the hint reflects the updated translation logic from _translate_processed_part
            translation_hint = f'"If {translated_body_hint}, then {translated_head_hint}."' # Logic in _translate_processed_part is updated

            # Format for standard rule
            return f"rule: {original_rule_formatted} :: [if] {processed_body} [then] {processed_head} :: {translation_hint}"
        except Exception as e:
            # Log error
            # print(f"Error transforming standard rule '{item_string}': {e}")
            return item_string # Fallback

    # Check if it's a strongly negated fact (starts with - but not :-)
    elif item_string_stripped.startswith("-"):
        try:
            original_fact = item_string.strip()
            # Ensure original ends with a period
            original_fact_formatted = original_fact if original_fact.endswith('.') else original_fact + '.'
            # Extract base fact without '-' and '.'
            base_fact = original_fact.lstrip('-')
            if base_fact.endswith('.'):
                base_fact = base_fact[:-1]
            # Format the output, using specific translation format
            transformed_part = f"<strong negation> {base_fact}"
            translation_format = f'"{base_fact} is explicitly false."' # Updated translation hint
            return f"fact: {original_fact_formatted} :: {transformed_part} :: {translation_format}"
        except Exception as e:
            # Log error (using logger if available, or print)
            # print(f"Error transforming strongly negated fact '{item_string}': {e}")
            return item_string # Fallback

    # Otherwise, assume it's a positive fact or query
    else:
        # Assume it's a positive fact (or query, though queries might need different handling later)
        try:
            original_fact = item_string.strip()
            # Ensure original ends with a period
            original_fact_formatted = original_fact if original_fact.endswith('.') else original_fact + '.'
            # Extract base fact without '.'
            base_fact = original_fact
            if base_fact.endswith('.'):
                base_fact = base_fact[:-1]
            # Format the output, using specific translation format
            transformed_part = base_fact # Positive facts don't have markers like <strong negation>
            translation_format = f'"{base_fact} is true."'
            # Check if it contains ':-' which would indicate it's actually a rule that failed the first check (unlikely but safe)
            if ":-" in original_fact:
                 # print(f"Warning: Item '{item_string}' looks like a rule but wasn't caught initially. Returning original.")
                 return item_string # Fallback for safety
            # Handle queries separately? For now, treat like facts for transformation output.
            # If it ends with '?', it's likely a query.
            if original_fact_formatted.endswith('?'):
                 # Queries don't typically get a translation hint in this format, return simpler structure?
                 # Or maybe return original? For now, keep consistent fact format.
                 pass # Keep fact format for now
            return f"fact: {original_fact_formatted} :: {transformed_part} :: {translation_format}"
        except Exception as e:
            # Log error
            # print(f"Error transforming positive fact '{item_string}': {e}")
            return item_string # Fallback


class TextulizationFramework:
    """
    用于处理事实状态查询任务的框架类。
    加载数据，调用大模型进行推理，并处理结果。
    """
    def __init__(self, model_name: str, batch_size: int = 20):
        """
        初始化框架。

        :param model_name: 用于实例化 CustomLLM 的模型名称。
        :type model_name: str
        :param batch_size: 每次调用 LLM 进行文本化的条目数量。
        :type batch_size: int
        """
        # --- Logger Setup ---
        # Get a logger specific to this class instance.
        # Configuration (level, handlers) should be done externally,
        # either by basicConfig in the main script or by the importing module's logging setup.
        # This logger will inherit level and handlers from its parent loggers unless configured directly.
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Logger obtained. Configuration depends on external setup.") # Log confirmation

        # --- Initialize other attributes ---
        self.batch_size = batch_size
        self.logger.info(f"Textulization batch size set to: {self.batch_size}")
        try:
            self.parser = JsonOutputParser()
            self.llm = CustomLLM(model_name)
            self.llm_free_talk = CustomLLM(model_name)
            self.llm_chain = self.llm | self.parser

            self.gpt_4o_mini = CustomLLM('mmm_gpt_4o_mini')
            self.chain_to_json = self.gpt_4o_mini | self.parser
            self.logger.info(f"CustomLLM initialized with model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing CustomLLM with model {model_name}: {e}", exc_info=True)
            # Decide how to handle LLM initialization failure. Raise error or set self.llm to None?
            # For now, let's raise the error to make the problem explicit.
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
            # read_jsonl raises FileNotFoundError directly
            self.logger.error(f"Error: File not found at {file_path}")
        except (_JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e: # Catch potential decode errors from orjson/json
             # read_jsonl raises specific decode errors with context
             self.logger.error(f"Error decoding JSON in {file_path}: {e}")
        except JsonUtilsError as e:
            # Catch custom errors from json_utils
            self.logger.error(f"An error occurred using json_utils: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            self.logger.exception(f"An unexpected error occurred while loading data: {e}") # Use exception for stack trace
        return data

    def run(self, data_item: Dict[str, Any]): # Updated return type annotation
        """
        处理单个数据项，并返回查询结果的状态。

        :param data_item: 单个数据项的字典。
        :type data_item: Dict[str, Any]
        :return: 包含文本化事实和规则的新数据项字典，如果出错则返回 None。
        :rtype: Dict[str, Any] | None
        """
        try:
            # --- Create a deep copy to avoid modifying the original item ---
            new_data_item = copy.deepcopy(data_item)
            item_id = new_data_item.get('id', 'N/A') # Get item ID for logging

            # --- Identify data structure and extract items for textualization ---
            all_symbolic_items = []
            original_data_structure = {} # Store original lists/dicts for later update

            if 'asp_program_dlv2' in new_data_item:
                # Structure 1: Contains asp_program_dlv2
                self.logger.debug(f"Item {item_id}: Detected structure 1 (asp_program_dlv2)")
                asp_program = new_data_item['asp_program_dlv2']
                original_data_structure['noiseless_facts'] = asp_program.get('noiseless_facts', [])
                original_data_structure['noiseless_rules'] = asp_program.get('noiseless_rules', [])
                original_data_structure['noisy_facts'] = asp_program.get('noisy_facts', [])
                original_data_structure['min_fact_dicts_for_query'] = asp_program.get('min_fact_dicts_for_query', [])
                original_data_structure['noisy_rules'] = asp_program.get('noisy_rules', {})
                original_data_structure['target_query'] = new_data_item.get('target_query', '') # Still might be relevant for context

                all_symbolic_items.extend(original_data_structure['noiseless_facts'])
                all_symbolic_items.extend(original_data_structure['noiseless_rules'])
                all_symbolic_items.extend(original_data_structure['noisy_facts'])
                all_symbolic_items.extend(original_data_structure['min_fact_dicts_for_query'])
                for rule_list in original_data_structure['noisy_rules'].values():
                    all_symbolic_items.extend(rule_list)
                structure_type = 1

            elif 'facts' in new_data_item and 'rules' in new_data_item:
                # Structure 2 or 3: Top-level facts, rules, answer_sets, maybe incorrect_answer_sets
                self.logger.debug(f"Item {item_id}: Detected structure 2/3 (top-level facts/rules)")
                original_data_structure['facts'] = new_data_item.get('facts', [])
                original_data_structure['rules'] = new_data_item.get('rules', [])
                original_data_structure['answer_sets'] = new_data_item.get('answer_sets', [])
                original_data_structure['incorrect_answer_sets'] = new_data_item.get('incorrect_answer_sets') # Get value or None
                original_data_structure['target_query'] = new_data_item.get('target_query', '') # Still might be relevant for context

                all_symbolic_items.extend(original_data_structure['facts'])
                all_symbolic_items.extend(original_data_structure['rules'])
                # Flatten answer_sets
                all_symbolic_items.extend(itertools.chain.from_iterable(original_data_structure.get('answer_sets', [])))

                # Determine primary structure type based on presence of options or incorrect sets
                if 'options' in new_data_item:
                    self.logger.debug(f"Item {item_id}: Detected structure 4 (answerset selection with options)")
                    original_data_structure['options'] = new_data_item.get('options', [])
                    # Flatten answer_sets within options for textualization
                    for option in original_data_structure['options']:
                        if isinstance(option, dict) and 'answer_set' in option:
                            all_symbolic_items.extend(option['answer_set'])
                    structure_type = 4
                elif original_data_structure.get('incorrect_answer_sets') is not None:
                    self.logger.debug(f"Item {item_id}: Detected structure 2 (answerset generation - no options)")
                    # Note: incorrect_sets processing happens below, regardless of type 2 or 4
                    structure_type = 2
                else:
                    self.logger.debug(f"Item {item_id}: Detected structure 3 (fact state querying - no options or incorrect sets)")
                    structure_type = 3

                # --- Always collect items from incorrect_answer_sets if present (for types 2 and 4) ---
                incorrect_sets = original_data_structure.get('incorrect_answer_sets')
                if incorrect_sets is not None:
                    self.logger.debug(f"Item {item_id}: Collecting items from incorrect_answer_sets (structure type {structure_type})")
                    # Handle both list of lists and list of dicts format
                    if incorrect_sets and isinstance(incorrect_sets[0], list): # Old format: list of lists
                        all_symbolic_items.extend(itertools.chain.from_iterable(incorrect_sets))
                    elif incorrect_sets and isinstance(incorrect_sets[0], dict): # New format: list of dicts
                        for item_dict in incorrect_sets:
                            if 'answer_set' in item_dict:
                                all_symbolic_items.extend(item_dict['answer_set'])
                # --- End collection from incorrect_answer_sets ---

                # --- NEW: Check and add items from answer_set_decision ---
                if 'answer_set_decision' in new_data_item and isinstance(new_data_item['answer_set_decision'], dict) and 'answerset' in new_data_item['answer_set_decision']:
                    self.logger.debug(f"Item {item_id}: Including items from answer_set_decision.")
                    original_data_structure['answer_set_decision'] = new_data_item['answer_set_decision'] # Store for later update
                    all_symbolic_items.extend(original_data_structure['answer_set_decision']['answerset'])
                # --- END NEW ---
            else:
                self.logger.error(f"Item {item_id}: Unknown data structure. Cannot determine fields for textualization.")
                return None # Cannot proceed

            # --- Remove duplicates while preserving order ---
            seen = set()
            unique_symbolic_items = [x for x in all_symbolic_items if x and not (x in seen or seen.add(x))] # Added check for non-empty x

            if not unique_symbolic_items:
                self.logger.warning(f"No unique non-empty symbolic items found to textualize for item {item_id}")
                return new_data_item # Return unmodified item

            # --- Prepare Prompt Context (Generic, using available fields) ---
            # Note: The prompt template might need adjustment if it strictly relies on asp_program_dlv2 structure
            template_string = prompts_textulization.get('textulization', {}).get('prompt')
            if not template_string:
                self.logger.error(f"Error: Prompt template 'textulization' not found in prompts.")
                return None

            jinja_env = Environment(loader=BaseLoader())
            template = jinja_env.from_string(template_string)

            # --- Build Simplified Prompt Context (Facts & Rules only, plus sets/query if present) ---
            # This context is primarily for the initial prompt rendering, not batch processing.
            # Batch processing uses only the items in the batch.
            simplified_render_context = {}
            # ... (rest of context building logic remains the same) ...
            aggregated_facts = []
            aggregated_rules = []

            if structure_type == 1:
                aggregated_facts.extend(original_data_structure.get('noiseless_facts', []))
                aggregated_facts.extend(original_data_structure.get('noisy_facts', []))
                aggregated_facts.extend(original_data_structure.get('min_fact_dicts_for_query', []))
                aggregated_rules.extend(original_data_structure.get('noiseless_rules', []))
                for rule_list in original_data_structure.get('noisy_rules', {}).values():
                    aggregated_rules.extend(rule_list)
            elif structure_type in [2, 3, 4]: # Include structure 4 here
                aggregated_facts.extend(original_data_structure.get('facts', []))
                aggregated_rules.extend(original_data_structure.get('rules', []))
                # Add sets/options if they exist for context rendering
                if original_data_structure.get('answer_sets'):
                    simplified_render_context['answer_sets'] = original_data_structure['answer_sets']
                if structure_type == 2 and original_data_structure.get('incorrect_answer_sets'):
                    simplified_render_context['incorrect_answer_sets'] = original_data_structure['incorrect_answer_sets']
                if structure_type == 4 and original_data_structure.get('options'):
                    simplified_render_context['options'] = original_data_structure['options']

            # Add aggregated facts and rules to the context *only* if they are not empty
            if aggregated_facts:
                simplified_render_context['facts'] = [f for f in aggregated_facts if f] # Filter empty
            if aggregated_rules:
                simplified_render_context['rules'] = [r for r in aggregated_rules if r] # Filter empty


            # --- Process in batches ---
            combined_results_json = {} # Stores all textualized results {symbolic: textual}
            num_items = len(unique_symbolic_items)
            num_batches = math.ceil(num_items / self.batch_size)
            self.logger.info(f"Item {item_id}: Processing {num_items} unique symbolic items in {num_batches} batches.")

            # --- 收集所有请求的规范化键，用于后续检查 ---
            all_requested_normalized_keys = set()

            for i in range(num_batches):
                start_index = i * self.batch_size
                end_index = min((i + 1) * self.batch_size, num_items)
                batch_items = unique_symbolic_items[start_index:end_index]
                self.logger.debug(f"Batch {i+1}/{num_batches}: Processing items {start_index} to {end_index-1}")

                if not batch_items:
                    continue

                # --- Transform rules and facts in batch items for the LLM prompt ---
                # Use the updated _transform_item_for_llm which includes the hint
                processed_batch_items_for_prompt = [_transform_item_for_llm(item) for item in batch_items]
                self.logger.debug(f"Batch {i+1}: Transformed items for prompt using _transform_item_for_llm (with hints).")

                # --- Use ORIGINAL items for Pydantic model keys ---
                field_definitions_batch = {}
                # Use _normalize_symbolic_key on ORIGINAL items to generate Pydantic model keys
                unique_keys_batch = {_normalize_symbolic_key(item) for item in batch_items} # Keys based on original items
                all_requested_normalized_keys.update(unique_keys_batch) # Collect all requested original keys

                for key in unique_keys_batch:
                     # 查找一个与规范化键匹配的原始条目，用于描述
                     original_like = next((item for item in batch_items if _normalize_symbolic_key(item) == key), key)
                     field_definitions_batch[key] = (str, Field(..., description=f"Textual representation of '{original_like}'"))

                # Dynamically create the Pydantic model for the current batch's expected output
                batch_model_name = f'translate_symbolic_to_textual_batch_{i+1}'
                batch_json_llm = None # Initialize outside try
                batch_chain_to_json = None
                try:
                    batch_response_format = create_model(batch_model_name, **field_definitions_batch)
                    # --- Create a new LLM instance and chain for this batch ---
                    batch_json_llm = CustomLLM('mmm_gpt_4o_mini') # Create new instance
                    batch_json_llm.response_format = batch_response_format # Set format on the new instance
                    batch_chain_to_json = batch_json_llm | self.parser # Create new chain
                    self.logger.debug(f"Batch {i+1}: Created new JSON LLM instance and chain with response format for keys: {list(unique_keys_batch)}")
                except Exception as e:
                    self.logger.error(f"Error creating Pydantic model or batch LLM for batch {i+1}: {e}", exc_info=True)
                    # Decide how to handle: skip batch, return None for item, etc.
                    # For now, let's log and potentially skip, leading to missing textualizations
                    continue # Skip this batch

                # --- Invoke LLM for the current batch ---
                try:
                    # 1. Render the initial prompt (contains general rules)
                    #    The context here might be less critical now hints are in the items
                    #    Pass the transformed items with hints for the LLM to see
                    batch_render_context = {'symbolic_items': processed_batch_items_for_prompt}
                    try:
                        # Render the Jinja template with the TRANSFORMED items (including hints)
                        batch_final_prompt_transformed = template.render(render_context=batch_render_context)
                        self.logger.debug(f"Batch {i+1}: Rendered prompt for {len(processed_batch_items_for_prompt)} transformed items (with hints).")
                    except Exception as render_e:
                        self.logger.error(f"Item {item_id}, Batch {i+1}: Error rendering Jinja template: {render_e}. Context keys: {list(batch_render_context.keys())}", exc_info=True)
                        continue # Skip this batch if template rendering fails

                    # 2. Generate free-form text using the prompt with TRANSFORMED items (with hints)
                    self.logger.debug(f"Batch {i+1}: Invoking free talk LLM with transformed items (with hints).")
                    results_free_batch: str = self.llm_free_talk.invoke(batch_final_prompt_transformed)
                    self.logger.debug(f"Batch {i+1}: Free talk response received.")

                    # 3. Ask for JSON based on the batch-specific response format (using ORIGINAL keys)
                    #    Provide the transformed items (with hints) and the free text result.
                    original_items_list_str = "\n".join([f"$ {item} $" for item in batch_items]) # Original items for reference in prompt
                    transformed_items_list_str = "\n".join([f"$ {item} $" for item in processed_batch_items_for_prompt]) # Transformed items (with hints) shown to LLM
                    target_keys_str = ', '.join(sorted(unique_keys_batch)) # Keys derived from original items

                    json_prompt = (
                        f"{batch_final_prompt_transformed}" # Includes general rules
                        f"\n\nThe following symbolic items were processed (transformed with translation hints):\n{transformed_items_list_str}\n\n"
                        f"The generated natural language description based on the initial prompt and the items is:\n{results_free_batch}\n\n"
                        f"Now, create a JSON object containing the final textual representation for each of the **original** symbolic items listed below. "
                        f"Use the translation hints provided with each item (after the second '::') as a strong guide for the structure and phrasing. "
                        f"The keys of the JSON object MUST correspond to these original items after normalization (removing quotes and trailing periods). "
                        f"The required keys for this batch are: {target_keys_str}.\n\n"
                        f"Original symbolic items for this batch:\n{original_items_list_str}\n\n"
                        f"Ensure the output is a valid JSON object mapping the normalized original keys to their final textual representation."
                    )

                    self.logger.debug(f"Batch {i+1}: JSON prompt created targeting original keys. Invoking batch-specific JSON LLM.")
                    # Use the batch-specific chain (expecting original keys)
                    if batch_chain_to_json: # Ensure the chain was created successfully
                        results_json_batch: dict = batch_chain_to_json.invoke(json_prompt) # Invoke with the new JSON prompt
                        self.logger.debug(f"Batch {i+1}: JSON response received (expected original keys).")
                    else:
                        self.logger.error(f"Batch {i+1}: Skipping JSON LLM invocation because chain creation failed.")
                        results_json_batch = {} # Ensure it's an empty dict if chain failed

                    # --- Combine results ---
                    # combined_results_json expects keys based on original items, which results_json_batch should now have
                    combined_results_json.update(results_json_batch)
                    self.logger.debug(f"Batch {i+1}: Combined results updated with {len(results_json_batch)} items (using original keys).")

                    # No delay between batches as requested

                except Exception as llm_e:
                     # Log the specific batch number and item ID in the error message
                     self.logger.error(f"Error during LLM invocation for batch {i+1}/{num_batches} of item {item_id}: {llm_e}", exc_info=True)
                     # Consider adding more specific error handling or retry logic here if needed.
                     # For now, log and continue; this batch's items might not be textualized.


            # --- Replace original symbolic items with textualized versions based on structure ---
            self.logger.info(f"Item {item_id}: Finished processing batches. Total textualized items received: {len(combined_results_json)}")

            # --- Helper functions for textualization (Defined within run scope) ---
            def get_textualized(item: str, results: Dict[str, str]) -> str:
                """
                使用规范化键获取文本化版本，如果找不到则回退到原始版本。

                :param item: 原始符号条目。
                :type item: str
                :param results: 包含文本化结果的字典 (键应该是规范化后的)。
                :type results: Dict[str, str]
                :return: 文本化后的字符串或原始字符串。
                :rtype: str
                """
                if not item: return "" # Handle empty strings gracefully
                normalized_key = _normalize_symbolic_key(item)
                textualized_value = results.get(normalized_key)

                if textualized_value is None:
                    # 检查这个键是否是我们期望从 LLM 获取的
                    if normalized_key in all_requested_normalized_keys:
                        # 如果查找失败，抛出错误而不是返回原始条目
                        error_message = f"Textualization failed for original item: '{item}'. Normalized key '{normalized_key}' was requested but not found in LLM results."
                        self.logger.error(error_message) # 记录错误日志
                        raise ValueError(error_message) # 抛出异常
                    else:
                        # 如果这个 key 本来就不在请求列表里（理论上不应发生，但也处理一下），也当作错误
                        error_message = f"Textualization lookup skipped for original item: '{item}'. Normalized key '{normalized_key}' was not found in the set of requested keys for the batch."
                        self.logger.warning(error_message) # 记录警告
                        # 为了严格性，也抛出错误。
                        raise ValueError(error_message)
                        # 或者，如果认为这种情况可以容忍，可以返回原始值：
                        # return item
                else:
                    # Basic cleanup: remove potential surrounding quotes from LLM output
                    return textualized_value.strip().strip('"')


            # Ensure these are defined *after* the updated get_textualized
            def textualize_list(item_list: List[str], results: Dict[str, str]) -> List[str]:
                """Textualizes a flat list of items."""
                return [get_textualized(item, results) for item in item_list if item] # Filter empty

            def textualize_nested_list(nested_list: List[List[str]], results: Dict[str, str]) -> List[List[str]]:
                """Textualizes a list of lists of items."""
                return [[get_textualized(item, results) for item in sublist if item] for sublist in nested_list] # Filter empty

            def textualize_dict_of_lists(dict_of_lists: Dict[str, List[str]], results: Dict[str, str]) -> Dict[str, List[str]]:
                 """Textualizes lists within a dictionary."""
                 return {key: textualize_list(item_list, results) for key, item_list in dict_of_lists.items()}

            def textualize_incorrect_answer_set_list(incorrect_list: List[Dict[str, Any]], results: Dict[str, str]) -> List[Dict[str, Any]]:
                """Textualizes the 'answer_set' within each dict in the incorrect_answer_sets list."""
                textualized_list = []
                for item_dict in incorrect_list:
                    if isinstance(item_dict, dict) and 'answer_set' in item_dict:
                        new_dict = item_dict.copy() # Avoid modifying original
                        new_dict['answer_set'] = textualize_list(item_dict['answer_set'], results)
                        textualized_list.append(new_dict)
                    else:
                        # Handle unexpected format, maybe log a warning or skip
                        self.logger.warning(f"Skipping unexpected item format in incorrect_answer_sets: {item_dict}")
                        textualized_list.append(item_dict) # Append original if format is wrong
                return textualized_list

            def textualize_options_list(options_list: List[Dict[str, Any]], results: Dict[str, str]) -> List[Dict[str, Any]]:
                """Textualizes the 'answer_set' within each dict in the options list."""
                textualized_list = []
                for option_dict in options_list:
                     if isinstance(option_dict, dict) and 'answer_set' in option_dict:
                         new_dict = option_dict.copy() # Avoid modifying original
                         new_dict['answer_set'] = textualize_list(option_dict['answer_set'], results)
                         textualized_list.append(new_dict)
                     else:
                         self.logger.warning(f"Skipping unexpected item format in options: {option_dict}")
                         textualized_list.append(option_dict) # Append original if format is wrong
                return textualized_list


            # --- Update the new_data_item based on the original structure ---
            # (This part uses the helper functions defined above, which now use the updated get_textualized)
            if structure_type == 1:
                # Update fields within asp_program_dlv2
                asp_program_update = new_data_item['asp_program_dlv2']
                asp_program_update['noiseless_facts'] = textualize_list(original_data_structure.get('noiseless_facts', []), combined_results_json)
                asp_program_update['noiseless_rules'] = textualize_list(original_data_structure.get('noiseless_rules', []), combined_results_json)
                asp_program_update['noisy_facts'] = textualize_list(original_data_structure.get('noisy_facts', []), combined_results_json)
                asp_program_update['min_fact_dicts_for_query'] = textualize_list(original_data_structure.get('min_fact_dicts_for_query', []), combined_results_json)
                asp_program_update['noisy_rules'] = textualize_dict_of_lists(original_data_structure.get('noisy_rules', {}), combined_results_json)
            elif structure_type == 2: # Answerset Generation
                new_data_item['facts'] = textualize_list(original_data_structure.get('facts', []), combined_results_json)
                new_data_item['rules'] = textualize_list(original_data_structure.get('rules', []), combined_results_json)
                new_data_item['answer_sets'] = textualize_nested_list(original_data_structure.get('answer_sets', []), combined_results_json)
                # Handle incorrect_answer_sets (could be list of lists or list of dicts)
                incorrect_sets = original_data_structure.get('incorrect_answer_sets')
                if incorrect_sets is not None:
                    if incorrect_sets and isinstance(incorrect_sets[0], list): # Old format
                        new_data_item['incorrect_answer_sets'] = textualize_nested_list(incorrect_sets, combined_results_json)
                    elif incorrect_sets and isinstance(incorrect_sets[0], dict): # New format
                        new_data_item['incorrect_answer_sets'] = textualize_incorrect_answer_set_list(incorrect_sets, combined_results_json)
                    else: # Empty list or unexpected format
                         new_data_item['incorrect_answer_sets'] = incorrect_sets # Keep as is
            elif structure_type == 3: # Fact State Querying
                new_data_item['facts'] = textualize_list(original_data_structure.get('facts', []), combined_results_json)
                new_data_item['rules'] = textualize_list(original_data_structure.get('rules', []), combined_results_json)
                new_data_item['answer_sets'] = textualize_nested_list(original_data_structure.get('answer_sets', []), combined_results_json)
                # No incorrect_answer_sets or options expected
            elif structure_type == 4: # Answerset Selection
                new_data_item['facts'] = textualize_list(original_data_structure.get('facts', []), combined_results_json)
                new_data_item['rules'] = textualize_list(original_data_structure.get('rules', []), combined_results_json)
                new_data_item['answer_sets'] = textualize_nested_list(original_data_structure.get('answer_sets', []), combined_results_json) # Textualize if present
                # Textualize incorrect_answer_sets (should be list of dicts)
                incorrect_sets = original_data_structure.get('incorrect_answer_sets')
                if incorrect_sets is not None:
                     new_data_item['incorrect_answer_sets'] = textualize_incorrect_answer_set_list(incorrect_sets, combined_results_json)
                # Textualize options
                options_list = original_data_structure.get('options')
                if options_list is not None:
                    new_data_item['options'] = textualize_options_list(options_list, combined_results_json)

                # --- NEW: Textualize answer_set_decision if present ---
                if 'answer_set_decision' in original_data_structure: # Check if it was extracted
                    decision_dict = original_data_structure['answer_set_decision']
                    if isinstance(decision_dict, dict) and 'answerset' in decision_dict:
                         new_decision_dict = decision_dict.copy() # Avoid modifying original stored structure
                         new_decision_dict['answerset'] = textualize_list(decision_dict['answerset'], combined_results_json)
                         new_data_item['answer_set_decision'] = new_decision_dict # Update the main item
                    else:
                         # Log warning if structure is unexpected but key exists
                         self.logger.warning(f"Item {item_id}: 'answer_set_decision' key exists but has unexpected format: {decision_dict}")
                         # Keep original if format is wrong? Or remove? Keep for now.
                         new_data_item['answer_set_decision'] = decision_dict
                # --- END NEW ---
            else:
                 # Should not happen due to earlier check, but log just in case
                 self.logger.error(f"Item {item_id}: Reached update stage with unknown structure type {structure_type}")


            # --- Create the textualization dictionary (maps original symbolic to textualized) ---
            textualization_dict = {}
            for item in unique_symbolic_items:
                # Calls the updated get_textualized defined above
                textualization_dict[item] = get_textualized(item, combined_results_json)

            # --- Add the textualization dictionary to the new data item ---
            new_data_item['textualization_dict'] = textualization_dict

            return new_data_item # Return the modified data item

        except KeyError as e:
            # Catch errors if expected keys are missing during processing
            self.logger.error(f"Error accessing data in data_item {data_item.get('id', 'N/A')}: Missing key {e}")
            return None
        except Exception as e:
            # Catch any other unexpected errors during the run method
            self.logger.exception(f"An unexpected error occurred during processing item {data_item.get('id', 'N/A')}: {e}") # Already uses exception
            return None

    def process_dataset(self, input_file_path: str, output_file_path: str, num_threads: int = 1, save_interval: int = 100):
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
        self.logger.info(f"Loading input data from: {input_file_path}")
        loaded_data = self.load_data(input_file_path)
        if not loaded_data:
            self.logger.error("Failed to load input data or input data is empty. Aborting processing.")
            return

        # --- 2. 加载已处理的 ID (如果输出文件存在) ---
        processed_ids: Set[str] = set()
        if os.path.exists(output_file_path):
            self.logger.info(f"Output file {output_file_path} exists. Loading processed IDs...")
            try:
                existing_data = read_jsonl(output_file_path)
                processed_ids = {item.get('id') for item in existing_data if item and 'id' in item}
                self.logger.info(f"Loaded {len(processed_ids)} processed IDs from existing output file.")
            except FileNotFoundError:
                 self.logger.warning(f"Output file {output_file_path} not found, starting fresh.") # Should not happen due to os.path.exists
            except (_JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e:
                 self.logger.error(f"Error decoding JSON in existing output file {output_file_path}. Consider backing up and deleting/fixing the file. Error: {e}")
                 return # Abort if output file is corrupted
            except JsonUtilsError as e:
                 self.logger.error(f"Error reading existing output file using json_utils: {e}")
                 return # Abort
            except Exception as e:
                 self.logger.exception(f"An unexpected error occurred while loading processed IDs: {e}")
                 return # Abort

        # --- 3. 过滤掉已处理项 ---
        items_to_process = [item for item in loaded_data if item.get('id') not in processed_ids]
        total_items_to_process = len(items_to_process)

        if not items_to_process:
            self.logger.info("No new items to process based on existing output file.")
            return

        self.logger.info(f"Starting dataset processing for {total_items_to_process} new items with {num_threads} threads.")
        self.logger.info(f"Results will be appended to: {output_file_path}")

        processed_count = 0
        successful_count = 0

        # --- 4. 确保输出目录存在 ---
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # --- 5. 使用多线程处理并写入文件 ---
        try:
            # Open file in append mode ('a')
            with open(output_file_path, 'a', encoding='utf-8') as outfile, \
                 concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

                # Submit jobs only for items that need processing
                future_to_item_id = {executor.submit(self.run, data_item): data_item.get('id', f'index_{i}')
                                     for i, data_item in enumerate(items_to_process)}

                # Process completed futures as they finish
                for future in tqdm(concurrent.futures.as_completed(future_to_item_id), total=total_items_to_process, desc="Processing dataset"):
                    item_id = future_to_item_id[future] # Get the original ID associated with the future
                    try:
                        result = future.result()
                        processed_count += 1
                        if result is not None:
                            # Serialize the result dictionary to a JSON string
                            # Use standard json.dumps to support ensure_ascii=False for non-ASCII chars
                            json_line = json.dumps(result, ensure_ascii=False)
                            # Write the JSON string as a line to the output file
                            outfile.write(json_line + '\n')
                            successful_count += 1

                            # Flush buffer periodically to ensure data is written
                            if processed_count % save_interval == 0:
                                outfile.flush()
                                self.logger.critical(f"Flushed output file after processing {processed_count} items.")

                    except Exception as exc:
                        processed_count += 1 # Count as processed even if failed
                        self.logger.error(f'Item {item_id} generated an exception during threaded execution: {exc}', exc_info=True)
                        # Do not write anything for failed items

                # Final flush after the loop finishes
                outfile.flush()
                self.logger.debug("Final flush of output file.")

        except IOError as e:
            self.logger.error(f"Error writing to output file {output_file_path}: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during dataset processing: {e}", exc_info=True)


        self.logger.info(f"Finished processing dataset. Attempted: {total_items_to_process}, Successfully processed and saved: {successful_count}.")
        # No return value needed as results are written directly

# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 1. 定义项目根目录 (假设此脚本位于 src/symtex_evaluation/ 下)
    # __file__ 是当前脚本的路径
    # os.path.dirname(__file__) 是脚本所在的目录 (src/symtex_evaluation)
    # os.path.dirname(os.path.dirname(__file__)) 是 src 目录
    # os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 是项目根目录
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # logging.info(f"Project path detected: {project_path}") # Use print or configure root logger before class init

    # --- Setup Root Logging (Optional but recommended if running as main) ---
    # Configure root logger *before* initializing the class if you want global settings.
    # The class logger (`src.dataset_generation.textulization.TextulizationFramework`) will inherit this level and handlers.
    # Set the desired level for the *entire application* when run as main here.
    # For example, use logging.DEBUG to see debug messages from the framework.
    logging.basicConfig(level=logging.CRITICAL , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logging.info(f"Project path detected: {project_path}") # Now logging works via root logger

    # 2. 配置模型名称和相对数据文件路径
    # mmm_gpt_41_mini_2025_04_14
    model_name_to_use = "local_qwen2_5_7b"
    # 定义相对于项目根目录的路径
    fname = 'a_symtex_task_answerset_selection_2025_04_27_15_46.jsonl'
    relative_jsonl_path = os.path.join('datasets', 'symtex_final', fname)
    relative_output_path = os.path.join('datasets', 'symtex_final_textual', fname)

    # 构建绝对路径
    jsonl_file_path = os.path.join(project_path, relative_jsonl_path)
    output_file_path = os.path.join(project_path, relative_output_path)
    logging.info(f"Using data file: {jsonl_file_path}")
    logging.info(f"Using output file: {output_file_path}")


    # 3. 初始化框架 (LLM 在内部初始化, Logger is configured inside __init__)
    try:
        framework = TextulizationFramework(model_name=model_name_to_use) # <--- 修改后的初始化，使用默认 batch_size
    except RuntimeError as e:
        logging.error(f"Framework initialization failed: {e}", exc_info=True)
        exit(1) # Exit if framework (and LLM) can't be initialized

    # 4. 处理数据
    # Set the number of threads for processing
    num_processing_threads = 1 # As requested
    save_interval_count = 1 # How often to flush the file buffer

    logging.info(f"Starting dataset processing with {num_processing_threads} threads. Save interval: {save_interval_count} items.")

    # 调用修改后的 process_dataset，传入文件路径
    framework.process_dataset(
        input_file_path=jsonl_file_path,
        output_file_path=output_file_path,
        num_threads=num_processing_threads,
        save_interval=save_interval_count
    )

    # 结果现在是直接写入文件的，不需要后续的保存步骤
    logging.info(f"Processing complete. Results (if any) are saved/appended in {output_file_path}")
