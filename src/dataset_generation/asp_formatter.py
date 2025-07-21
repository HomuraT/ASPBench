# src/dataset_generation/asp_formatter.py
from typing import Dict, List, Union, Optional, Any # Added Any
import logging # Added logging

def format_atom_dlv2(pred_dict: dict, idx_to_name: Dict[int, str], is_fact: bool = False, is_body: bool = False, var_idx_to_name: Optional[Dict[int, str]] = None) -> str:
    """
    Formats a predicate dictionary into a DLV2 atom string, optionally using custom variable names.

    :param pred_dict: Dictionary representing the predicate.
                      Expected keys: 'predicateIdx', 'variables',
                      Optional keys: 'strong negation', 'default negation'.
    :type pred_dict: dict
    :param idx_to_name: Mapping from predicate index to predicate name.
    :type idx_to_name: Dict[int, str]
    :param is_fact: Indicates if the atom is a fact (affects variable quoting). Defaults to False.
    :type is_fact: bool
    :param is_body: Indicates if the atom is in the body of a rule (affects default negation). Defaults to False.
    :type is_body: bool
    :param var_idx_to_name: Optional mapping from variable index (int) to desired variable name (str).
                            If provided, uses these names; otherwise defaults to "V<index>".
    :type var_idx_to_name: Optional[Dict[int, str]]
    :return: The formatted DLV2 atom string.
    :rtype: str
    """
    idx = pred_dict['predicateIdx']
    name = idx_to_name.get(idx, f"P{idx}") # Fallback name if index not in map
    variables = pred_dict['variables']
    strong_neg = pred_dict.get('strong negation', False)
    # Default negation only applies if it's explicitly true AND the atom is in the body
    default_neg = pred_dict.get('default negation', False) and is_body

    prefix = ""
    if default_neg:
        prefix += "not "
    if strong_neg:
        # Add space after '-' only if not preceded by 'not '
        prefix += "- " if not default_neg else "-"

    # Standardize variables and apply quotes for facts
    formatted_vars = []
    for v in variables:
        # Determine the base variable string representation
        if isinstance(v, int) and var_idx_to_name and v in var_idx_to_name:
            # Use custom name if available for integer index
            var_str_base = var_idx_to_name[v]
        elif isinstance(v, int):
            # Default format for integer index
            var_str_base = f"V{v}"
        else:
            # Use string representation for non-integer variables (should ideally not happen based on generation logic)
            var_str_base = str(v)

        # Apply quotes only for facts
        if is_fact:
            # Quote the determined base string if it's a fact
            formatted_vars.append(f'"{var_str_base}"')
        else:
            # Use the base string directly if not a fact
            formatted_vars.append(var_str_base)

    var_str = ", ".join(formatted_vars)
    if var_str:
        return f"{prefix}{name}({var_str})"
    else:
        # Handle atoms with no variables
        return f"{prefix}{name}"

def format_rule_dict_to_dlv2_lines(rule_dict: dict, idx_to_name: Dict[int, str], use_disjunction: bool = False, var_idx_to_name: Optional[Dict[int, str]] = None) -> List[str]:
    """
    Formats a rule dictionary into a list of DLV2 rule strings, optionally using custom variable names.
    Handles single-head rules directly.
    For multi-head rules:
    - If use_disjunction=True, formats as a single line using '|'.
    - If use_disjunction=False, generates a separate rule string for each head.

    :param rule_dict: Dictionary representing the rule.
                      Expected keys: 'head' (list of predicate dicts),
                      'body' (list of predicate dicts).
    :type rule_dict: dict
    :param idx_to_name: Mapping from predicate index to predicate name.
    :type idx_to_name: Dict[int, str]
    :param use_disjunction: If True, formats multi-head rules using '|' in a single string.
                           If False, generates multiple rule strings. Defaults to False.
    :type use_disjunction: bool
    :param var_idx_to_name: Optional mapping from variable index (int) to desired variable name (str).
    :type var_idx_to_name: Optional[Dict[int, str]]
    :return: A list of DLV2 rule strings.
    :rtype: List[str]
    :raises ValueError: If the rule has no head.
    """
    head_pred_dicts = rule_dict.get('head', [])
    body_pred_dicts = rule_dict.get('body', [])

    if not head_pred_dicts:
        raise ValueError("Rule dictionary must contain at least one 'head' predicate.")

    # Pass var_idx_to_name to format_atom_dlv2
    head_atoms = [format_atom_dlv2(p, idx_to_name, is_fact=False, is_body=False, var_idx_to_name=var_idx_to_name) for p in head_pred_dicts]
    body_atoms = [format_atom_dlv2(p, idx_to_name, is_fact=False, is_body=True, var_idx_to_name=var_idx_to_name) for p in body_pred_dicts]
    body_str = ", ".join(body_atoms)

    formatted_rules = []
    if len(head_atoms) == 1 or use_disjunction:
        # Single head or using disjunction for multiple heads
        head_str = " | ".join(head_atoms)
        rule_str = f"{head_str} :- {body_str}." if body_str else f"{head_str}."
        formatted_rules.append(rule_str)
    else:
        # Multiple heads and not using disjunction: split into multiple rules
        for head_atom in head_atoms:
            rule_str = f"{head_atom} :- {body_str}." if body_str else f"{head_atom}."
            formatted_rules.append(rule_str)

    return formatted_rules

def dict_to_asp_strings(data_dict: Dict, idx_to_name: Dict[int, str], is_fact: bool, use_disjunction_for_rules: bool = False, var_idx_to_name: Optional[Dict[int, str]] = None) -> List[str]:
    """
    Converts a dictionary representing an ASP fact or rule into a list of ASP (DLV2) strings,
    optionally using custom variable names.
    A single fact results in a list with one string.
    A rule can result in one or more strings depending on the number of heads and the 'use_disjunction_for_rules' flag.

    :param data_dict: The dictionary representing the fact or rule.
                      For facts, expected keys: 'predicateIdx', 'variables', optional 'strong negation'.
                      For rules, expected keys: 'head', 'body' (lists of predicate dicts).
    :type data_dict: Dict
    :param idx_to_name: Mapping from predicate index to predicate name.
    :type idx_to_name: Dict[int, str]
    :param is_fact: True if the dictionary represents a fact, False if it represents a rule.
    :type is_fact: bool
    :param use_disjunction_for_rules: For rules with multiple heads: use '|' in a single string if True;
                                      generate separate rules if False (default).
    :type use_disjunction_for_rules: bool
    :param var_idx_to_name: Optional mapping from variable index (int) to desired variable name (str).
    :type var_idx_to_name: Optional[Dict[int, str]]
    :return: A list containing the corresponding ASP code line(s).
    :rtype: List[str]
    :raises ValueError: If input rule dict has no head.
    """
    if is_fact:
        # Pass var_idx_to_name to format_atom_dlv2 for facts
        formatted_atom = format_atom_dlv2(data_dict, idx_to_name, is_fact=True, is_body=False, var_idx_to_name=var_idx_to_name)
        return [f"{formatted_atom}."]
    else:
        # Pass var_idx_to_name to format_rule_dict_to_dlv2_lines for rules
        return format_rule_dict_to_dlv2_lines(data_dict, idx_to_name, use_disjunction=use_disjunction_for_rules, var_idx_to_name=var_idx_to_name)


def format_dict_structure_to_asp(
    dict_structure: Dict[str, Any],
    idx_to_name: Dict[int, str],
    fact_var_idx_to_name: Optional[Dict[int, str]] = None,
    rule_var_idx_to_name: Optional[Dict[int, str]] = None,
    use_disjunction: bool = False
) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    """
    Converts a dictionary containing various ASP elements (facts, rules)
    into a dictionary of formatted ASP (DLV2) string lists.

    Handles potential errors during formatting by logging warnings and returning
    error comments within the string lists.

    :param dict_structure: Dictionary containing lists/dicts of facts and rules.
                           Expected keys like 'noiseless_facts', 'noiseless_rules',
                           'noisy_facts', 'noisy_rules', 'min_fact_dicts_for_query'.
    :type dict_structure: Dict[str, Any]
    :param idx_to_name: Mapping from predicate index to predicate name.
    :type idx_to_name: Dict[int, str]
    :param fact_var_idx_to_name: Optional mapping for fact variable names.
    :type fact_var_idx_to_name: Optional[Dict[int, str]]
    :param rule_var_idx_to_name: Optional mapping for rule variable names (e.g., V0, V1).
                                 If None, defaults are used by underlying formatters.
    :type rule_var_idx_to_name: Optional[Dict[int, str]]
    :param use_disjunction: Whether to use disjunction for multi-head rules.
    :type use_disjunction: bool
    :return: A dictionary where keys correspond to the input structure's keys,
             and values are lists of formatted ASP strings or error comments.
             'noisy_rules' will have a nested dictionary structure.
    :rtype: Dict[str, Union[List[str], Dict[str, List[str]]]]
    """
    asp_program_dlv2: Dict[str, Union[List[str], Dict[str, List[str]]]] = {}

    # Helper function to safely format a single element (fact or rule dict)
    def _safe_format_element(element_dict: Dict[str, Any], is_fact: bool) -> List[str]:
        var_map = fact_var_idx_to_name if is_fact else rule_var_idx_to_name
        try:
            # Ensure idx_to_name is not empty, provide a fallback if necessary
            current_idx_to_name = idx_to_name
            if not current_idx_to_name:
                temp_idx_to_name = {}
                if 'predicateIdx' in element_dict:
                     temp_idx_to_name[element_dict['predicateIdx']] = f"P{element_dict['predicateIdx']}"
                elif not is_fact and 'head' in element_dict and element_dict['head']:
                     for head_atom in element_dict['head']:
                         if 'predicateIdx' in head_atom:
                              idx = head_atom['predicateIdx']
                              temp_idx_to_name[idx] = f"P{idx}"
                # Add more checks if needed (e.g., body)
                current_idx_to_name = temp_idx_to_name # Use the temporary map

            asp_strings: List[str] = dict_to_asp_strings(
                element_dict,
                current_idx_to_name,
                is_fact=is_fact,
                use_disjunction_for_rules=use_disjunction,
                var_idx_to_name=var_map
            )
            return asp_strings if asp_strings else [f"% Error: Empty result formatting {element_dict}"]
        except ValueError as e:
            logging.warning(f"Formatting error for {'fact' if is_fact else 'rule'}: {element_dict}. Error: {e}")
            return [f"% Error formatting {'fact' if is_fact else 'rule'}: {e}"]
        except KeyError as e:
            logging.warning(f"Missing key during formatting {'fact' if is_fact else 'rule'}: {element_dict}. Error: {e}")
            return [f"% Error formatting {'fact' if is_fact else 'rule'} due to missing key: {e}"]
        except Exception as e:
            logging.warning(f"Unexpected error formatting {'fact' if is_fact else 'rule'}: {element_dict}. Error: {e}", exc_info=True)
            return [f"% Error formatting {'fact' if is_fact else 'rule'}: {e}"]

    # Process lists of fact dictionaries
    fact_keys = ['noiseless_facts', 'noisy_facts', 'min_fact_dicts_for_query']
    for key in fact_keys:
        if key in dict_structure and isinstance(dict_structure[key], list):
            formatted_list = []
            for fact_dict in dict_structure[key]:
                if isinstance(fact_dict, dict):
                    formatted_list.extend(_safe_format_element(fact_dict, is_fact=True))
                else:
                    logging.warning(f"Skipping non-dict item in '{key}': {fact_dict}")
                    formatted_list.append(f"% Error: Expected dict, got {type(fact_dict)}")
            asp_program_dlv2[key] = formatted_list

    # Process lists of rule dictionaries
    rule_keys = ['noiseless_rules']
    for key in rule_keys:
        if key in dict_structure and isinstance(dict_structure[key], list):
            formatted_list = []
            for rule_dict in dict_structure[key]:
                if isinstance(rule_dict, dict):
                    formatted_list.extend(_safe_format_element(rule_dict, is_fact=False))
                else:
                    logging.warning(f"Skipping non-dict item in '{key}': {rule_dict}")
                    formatted_list.append(f"% Error: Expected dict, got {type(rule_dict)}")
            asp_program_dlv2[key] = formatted_list

    # Process noisy_rules (dictionary of lists of rule dictionaries)
    if 'noisy_rules' in dict_structure and isinstance(dict_structure['noisy_rules'], dict):
        noisy_rules_formatted: Dict[str, List[str]] = {}
        for rule_type, rule_list in dict_structure['noisy_rules'].items():
            if isinstance(rule_list, list):
                formatted_list = []
                for rule_dict in rule_list:
                    if isinstance(rule_dict, dict):
                        formatted_list.extend(_safe_format_element(rule_dict, is_fact=False))
                    else:
                        logging.warning(f"Skipping non-dict item in noisy_rules['{rule_type}']: {rule_dict}")
                        formatted_list.append(f"% Error: Expected dict, got {type(rule_dict)}")
                noisy_rules_formatted[rule_type] = formatted_list
            else:
                 logging.warning(f"Expected list for noisy_rules['{rule_type}'], got {type(rule_list)}. Skipping.")
                 noisy_rules_formatted[rule_type] = [f"% Error: Expected list, got {type(rule_list)}"]
        asp_program_dlv2['noisy_rules'] = noisy_rules_formatted

    return asp_program_dlv2


# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    example_idx_to_name = {0: "p", 1: "q", 2: "r", 3: "s"} # Added 's' for variety
    example_var_map = {1: "Tom", 2: "Jerry", 3: "Spike", 4: "ads"} # Custom variable names

    # Example Fact
    fact_dict = {'predicateIdx': 0, 'variables': [1, 5], 'strong negation': False}
    print(f"Fact (Default Vars): {dict_to_asp_strings(fact_dict, example_idx_to_name, is_fact=True)}")
    # Expected: ['p("V1", "V2").']
    print(f"Fact (Custom Vars): {dict_to_asp_strings(fact_dict, example_idx_to_name, is_fact=True, var_idx_to_name=example_var_map)}")
    # Expected: ['p("Tom", "Jerry").']


    # Example Fact with Strong Negation
    fact_dict_neg = {'predicateIdx': 1, 'variables': [3], 'strong negation': True}
    print(f"Neg Fact (Default Vars): {dict_to_asp_strings(fact_dict_neg, example_idx_to_name, is_fact=True)}")
    # Expected: ['- q("V3").']
    print(f"Neg Fact (Custom Vars): {dict_to_asp_strings(fact_dict_neg, example_idx_to_name, is_fact=True, var_idx_to_name=example_var_map)}")
    # Expected: ['- q("Spike").']

    # Example Rule (Single Head) - Using integer variables
    rule_dict_single = {
        'head': [{'predicateIdx': 0, 'variables': [1], 'strong negation': False}], # Variable V1
        'body': [
            {'predicateIdx': 1, 'variables': [1], 'strong negation': False, 'default negation': False}, # Variable V1
            {'predicateIdx': 2, 'variables': [1], 'strong negation': True, 'default negation': True}   # Variable V1
        ]
    }
    print(f"Rule (Single, Default Vars): {dict_to_asp_strings(rule_dict_single, example_idx_to_name, is_fact=False)}")
    # Expected: ['p(V1) :- q(V1), not - r(V1).']
    print(f"Rule (Single, Custom Vars): {dict_to_asp_strings(rule_dict_single, example_idx_to_name, is_fact=False, var_idx_to_name=example_var_map)}")
    # Expected: ['p(Tom) :- q(Tom), not - r(Tom).']


    # Example Rule (Multi-Head, Disjunction=False - Now splits) - Using integer variables
    rule_dict_multi = {
        'head': [
            {'predicateIdx': 0, 'variables': [1], 'strong negation': False}, # Variable V1
            {'predicateIdx': 1, 'variables': [2], 'strong negation': False}  # Variable V2
        ],
        'body': [{'predicateIdx': 3, 'variables': [1, 2], 'strong negation': False}] # Variables V1, V2
    }
    print(f"Rule (Multi, No Dis, Default Vars): {dict_to_asp_strings(rule_dict_multi, example_idx_to_name, is_fact=False, use_disjunction_for_rules=False)}")
    # Expected: ['p(V1) :- s(V1, V2).', 'q(V2) :- s(V1, V2).']
    print(f"Rule (Multi, No Dis, Custom Vars): {dict_to_asp_strings(rule_dict_multi, example_idx_to_name, is_fact=False, use_disjunction_for_rules=False, var_idx_to_name=example_var_map)}")
    # Expected: ['p(Tom) :- s(Tom, Jerry).', 'q(Jerry) :- s(Tom, Jerry).']


    # Example Rule (Multi-Head, Disjunction=True) - Using integer variables
    print(f"Rule (Multi, Dis, Default Vars): {dict_to_asp_strings(rule_dict_multi, example_idx_to_name, is_fact=False, use_disjunction_for_rules=True)}")
    # Expected: ['p(V1) | q(V2) :- s(V1, V2).']
    print(f"Rule (Multi, Dis, Custom Vars): {dict_to_asp_strings(rule_dict_multi, example_idx_to_name, is_fact=False, use_disjunction_for_rules=True, var_idx_to_name=example_var_map)}")
    # Expected: ['p(Tom) | q(Jerry) :- s(Tom, Jerry).']


    # Example Rule with no body
    rule_dict_no_body = {
        'head': [{'predicateIdx': 2, 'variables': [], 'strong negation': False}], # No variables
        'body': []
    }
    print(f"Rule (No Body): {dict_to_asp_strings(rule_dict_no_body, example_idx_to_name, is_fact=False)}")
    # Expected: ['r.'] (No variables to map)

    # --- Example for format_dict_structure_to_asp ---
    print("\n--- Testing format_dict_structure_to_asp ---")
    example_structure = {
        'noiseless_facts': [
            {'predicateIdx': 0, 'variables': [1, 5], 'strong negation': False},
            {'predicateIdx': 1, 'variables': [3], 'strong negation': True}
        ],
        'noiseless_rules': [
            {
                'head': [{'predicateIdx': 0, 'variables': [1], 'strong negation': False}],
                'body': [
                    {'predicateIdx': 1, 'variables': [1], 'strong negation': False, 'default negation': False},
                    {'predicateIdx': 2, 'variables': [1], 'strong negation': True, 'default negation': True}
                ]
            }
        ],
        'noisy_facts': [
             {'predicateIdx': 3, 'variables': [4], 'strong negation': False}
        ],
        'noisy_rules': {
            'type1': [
                {
                    'head': [{'predicateIdx': 1, 'variables': [2], 'strong negation': False}],
                    'body': [{'predicateIdx': 3, 'variables': [1, 2], 'strong negation': False}]
                }
            ],
            'type2': [] # Empty list example
        },
        'min_fact_dicts_for_query': [
            {'predicateIdx': 1, 'variables': [3], 'strong negation': True}
        ],
        'other_data': "should be ignored" # Example of extra data
    }

    formatted_asp = format_dict_structure_to_asp(
        example_structure,
        example_idx_to_name,
        fact_var_idx_to_name=example_var_map, # Use custom names for facts
        rule_var_idx_to_name=None, # Use default V0, V1 for rules
        use_disjunction=False
    )

    import json
    print(json.dumps(formatted_asp, indent=2))
    # Expected output structure (with actual formatted strings):
    # {
    #   "noiseless_facts": [
    #     "p(\"Tom\", \"V5\").",  <-- Note: V5 because 5 is not in example_var_map
    #     "- q(\"Spike\")."
    #   ],
    #   "noiseless_rules": [
    #     "p(V1) :- q(V1), not - r(V1)."
    #   ],
    #   "noisy_facts": [
    #      "s(\"ads\")."
    #   ],
    #   "noisy_rules": {
    #     "type1": [
    #       "q(V2) :- s(V1, V2)."
    #     ],
    #     "type2": []
    #   },
    #   "min_fact_dicts_for_query": [
    #     "- q(\"Spike\")."
    #   ]
    # }
