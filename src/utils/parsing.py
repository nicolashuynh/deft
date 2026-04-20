import json
import logging
from copy import copy

from src.utils.feature import FeatureInfo

logger = logging.getLogger(__name__)


def _parse_items_for_prompt(prompt_items, parse_item_fn):
    """Parse all generations for one prompt with shared append-on-success logic."""
    parsed_items = []
    for prompt_item in prompt_items:
        parsed_item = parse_item_fn(prompt_item)
        if parsed_item is not None:
            parsed_items.append(parsed_item)
    return parsed_items


def extract_generations(list_responses):
    """Extract generations."""
    list_generations_all_prompts = []
    indices_extracted = []
    total_token_used = 0

    for i, response in enumerate(list_responses):
        if response is not None:
            # Update token usage
            print(type(response))
            print(response)
            token_used = response["usage"]["total_tokens"]
            total_token_used += token_used

            try:
                # Parse the equations
                list_generations = parsing_end2end(response)
                list_generations_all_prompts.append(list_generations)
                indices_extracted.append(i)

            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
    assert len(list_generations_all_prompts) == len(
        indices_extracted
    ), f"The number of extracted generations should be the same as the number of indices extracted, but got {len(list_generations_all_prompts)} generations vs {len(indices_extracted)} indices extracted."
    return list_generations_all_prompts, indices_extracted, total_token_used


def parse_code_feature(list_responses, list_features):
    # We use the list_features to add the function at the end of the FeatureInfo object
    """Parse code feature."""
    assert len(list_responses) == len(
        list_features
    ), f"The number of responses and features should be the same, but got {len(list_responses)} responses and {len(list_features)} features."

    full_list_code_str, indices_extracted, total_token_used = extract_generations(
        list_responses
    )
    list_feature_all_prompts = []

    for list_code, prompt_index in zip(full_list_code_str, indices_extracted):
        source_feature = list_features[prompt_index]

        def parse_code_item(code_str):
            """Parse code item."""
            cleaned_code = clean_code_string(code_str)

            try:
                converted_fn = convert_code_to_fn(cleaned_code)
                updated_feature = copy(source_feature)
                updated_feature.fn = converted_fn
                updated_feature.string = cleaned_code
                return updated_feature
            except Exception as e:
                logger.error(f"Failed to convert the code to a Python function: {e}")
                logger.error(f"The clean code was: {cleaned_code}")
                logger.info(f"The code is: {code_str}")
                return None

        list_feature = _parse_items_for_prompt(list_code, parse_code_item)
        list_feature_all_prompts.append(list_feature)

    return list_feature_all_prompts, total_token_used


def clean_code_string(code_str):
    """Clean the code string by extracting everything from the first 'def' keyword.
    """
    if "def " not in code_str:
        raise ValueError("No function definition found in code string")

    # Extract everything from the first 'def'
    code = code_str[code_str.index("def ") :]

    # Remove any trailing backticks or whitespace
    code = code.strip("`").strip()

    return code


def parse_proposed_feature(list_responses, output_rationale=True):
    """Parse the LLM reponses obtained via an async call.
    """
    full_list_features, _, total_token_used = extract_generations(list_responses)
    # Note that full_list_features is a list of lists (length of outer list: number of prompts, lengths of inner lists: number of generations per prompt)

    list_converted_features_all_prompts = []
    for list_proposed_feature_str in full_list_features:

        def parse_proposed_item(proposed_feature_str):
            """Parse proposed item."""
            logger.info(f"Proposed feature: {proposed_feature_str}")
            cleaned_json_str = clean_json_string(proposed_feature_str)

            try:
                # Convert the json to an actual dictionary
                converted_dic = json.loads(cleaned_json_str)

                logger.info(f"Converted dictionary: {converted_dic}")
                # Create a FeatureInfo object
                if output_rationale:
                    converted_feature = FeatureInfo(
                        fn=None,
                        name=converted_dic["name"],
                        description=converted_dic["description"],
                        rationale=converted_dic["rationale"],
                    )
                else:
                    # If no rationale is provided, we can still create the FeatureInfo object
                    # but we will not include the rationale field
                    # This is useful for cases where the LLM does not provide a rationale
                    converted_feature = FeatureInfo(
                        fn=None,
                        name=converted_dic["name"],
                        description=converted_dic["description"],
                        rationale="",
                    )

                return converted_feature
            except Exception as e:
                logger.error(f"Failed to convert the proposed feature: {e}")
                return None

        list_converted_features = _parse_items_for_prompt(
            list_proposed_feature_str, parse_proposed_item
        )
        list_converted_features_all_prompts.append(list_converted_features)

    return list_converted_features_all_prompts, total_token_used


def clean_json_string(json_str):
    """Clean the JSON string by extracting between the first '{' and last '}'.
    """
    if "{" not in json_str or "}" not in json_str:
        return json_str

    # Extract from the first '{' to the last '}'
    start = json_str.index("{")
    end = json_str.rindex("}") + 1  # +1 to include the closing brace
    return json_str[start:end]


def parsing_end2end(response):
    """Handle parsing end2end."""
    list_generations = []
    n_processes = len(response["choices"])

    for process_i in range(n_processes):

        data = response["choices"][process_i]["message"]["content"]
        if data.startswith("##"):
            data = data[2:]
        if data.endswith("##"):
            data = data[:-2]

        list_generations.append(data)

    return list_generations


def convert_code_to_fn(code_str):
    """Compile string containing Python function code into callable function."""
    # Add safety checks
    if "exec" in code_str or "eval" in code_str:
        raise ValueError("Unsafe code detected")

    # Create namespace for function
    namespace = {}

    try:
        # Execute the code in isolated namespace
        exec(code_str, namespace)

        # Get function name (assumes single function in code)
        func_name = [name for name, obj in namespace.items() if callable(obj)][0]

        return namespace[func_name]

    except Exception as e:
        raise ValueError(f"Failed to compile function: {str(e)}")
