# Standard library imports
from typing import Any, Dict, List, Optional, Type, Union

# Third-party imports
from langchain.llms.base import LLM
from langchain_core.messages import AIMessage
from openai import OpenAI  # Keep OpenAI if call_openai_api uses it directly, but select_call might return others
from pydantic import BaseModel, Field
from langchain_core.language_models.llms import LLM  # Re-add this specific import
from langchain_core.outputs import Generation, GenerationChunk, LLMResult, ChatGeneration  # Re-add these
from langchain_core.callbacks.manager import CallbackManagerForLLMRun  # Re-add this

# Removed ChatOpenAI imports as they are not used in this version
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Local application imports
from apiHelper.apis.api_selector import select_call_api  # Re-add this
from apiHelper.managers.db import get_model_and_api_info  # Used in __init__
from apiHelper.utils.messages import LLMMessageBuilder  # Re-add this


class CustomLLM(LLM):
    config: Optional[dict] = Field(None, exclude=True)
    base_url: Optional[str] = Field(None, exclude=True)
    api_key: Optional[str] = Field(None, exclude=True)
    response_format: Optional[Union[Type[BaseModel], dict]] = Field(None, exclude=True)
    model_name: Optional[str] = Field(None, exclude=True)
    tools: Optional[List[Dict[str, Any]]] = Field(None, exclude=True)  # Renamed from functions
    model_params: Optional[Dict[str, Any]] = Field(default_factory=dict,
                                                   description="Additional parameters for the model API call.",
                                                   exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model_config_name, **kwargs: Any):
        super().__init__(**kwargs)
        config = get_model_and_api_info(model_config_name)
        self.base_url = config['base_url']
        self.api_key = config['api_key']
        self.model_name = config['model_name']
        self.model_params = config.get('params', {})

    # Re-add _prepare_prompt method (or ensure the logic is inside _call/_generate)
    def _prepare_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Prepare the prompt for the model, ensuring it's in the required message list format.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The input prompt.

        Returns:
            List[Dict[str, str]]: The prompt formatted as a list of message dictionaries.
        """
        if isinstance(prompt, str):
            _messages = LLMMessageBuilder()
            _messages.create_messages_from_text(prompt)
            messages_list = _messages.to_messages()
        elif isinstance(prompt, list):
            messages_list = prompt
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}. Expected str or List[Dict[str, str]].")
        return messages_list

    # Restore the previous _call method implementation
    def _call(self, prompt: Union[str, List[Dict[str, str]]], stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Core logic for single LLM calls, getting full response and extracting content.

        Args:
            prompt (Union[str, List[Dict[str, str]]]): The prompt to pass to the model.
            stop (Optional[List[str]]): Optional list of stop sequences.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            str: The generated text response (content only).

        Raises:
            ValueError: If the underlying API call fails or returns unexpected data.
        """
        messages = self._prepare_prompt(prompt)
        call_function = select_call_api(self.model_name)

        request_params = {**self.model_params, **kwargs, "stream": False}
        if stop:
            request_params['stop'] = stop

        try:
            response_data = call_function(
                messages,
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name,
                response_format=self.response_format,
                tools=self.tools,  # Pass tools instead of functions
                return_full_completion=True,  # Get the full object
                **request_params
            )

            all_reasoning_content = []
            text_content = ""
            # Extract content and reasoning from response
            if hasattr(response_data, 'choices') and response_data.choices:
                # Iterate through all choices to find reasoning content
                for i, choice in enumerate(response_data.choices):
                    if hasattr(choice, 'message') and choice.message:
                        reasoning = getattr(choice.message, 'reasoning_content', None)
                        if reasoning:
                            all_reasoning_content.append(f"<think>\n{reasoning}\n</think>")

                        # Use the content from the first choice as the main answer
                        if i == 0:
                            text_content = choice.message.content or ""

                # Fallback for Anthropic-like structure if needed (less likely for reasoning)
                if not text_content and hasattr(response_data, 'content') and isinstance(response_data.content,
                                                                                         list) and response_data.content:
                    first_content_block = response_data.content[0]
                    if hasattr(first_content_block, 'text'):
                        text_content = first_content_block.text or ""

            # Combine reasoning (if any) and the main answer content
            final_output = ""
            if all_reasoning_content:
                final_output += "\n".join(all_reasoning_content) + "\n\n"  # Separate reasoning blocks

            final_output += text_content  # Append the main answer

            return final_output

        except Exception as e:
            raise ValueError(f"API call failed in _call: {e}") from e

    @property
    def _identifying_params(self) -> dict:
        return {"api_url": self.base_url, "model_params": self.model_params, "response_format": self.response_format}

    @property
    def _llm_type(self) -> str:
        return self.model_name

    # Restore the _generate method with detailed token counting
    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        """
        Core logic for LLM generation, getting full response and detailed token usage.
        Handles both text generation and tool calls (returns choice JSON if tool call).
        Includes prepending reasoning_content.
        """
        generations = []
        aggregated_token_usage = {}
        call_function = select_call_api(self.model_name)

        for prompt_text in prompts:
            messages = self._prepare_prompt(prompt_text)
            current_request_params = {**self.model_params, **kwargs}
            if stop:
                current_request_params['stop'] = stop
            current_request_params.pop('stream', None)

            try:
                response_data: Any = call_function(
                    messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model_name=self.model_name,
                    response_format=self.response_format,
                    tools=self.tools,
                    tool_choice=None,  # Let the underlying API handle default/passed tool_choice from kwargs
                    return_str=False,
                    return_full_completion=True,
                    **current_request_params
                )

                if not hasattr(response_data, 'choices') or not response_data.choices:
                    raise ValueError(
                        f"API response did not contain expected choices. Response type: {type(response_data)}")

                first_choice = response_data.choices[0]
                message_of_first_choice = first_choice.message
                finish_reason_of_first_choice = first_choice.finish_reason

                generation_info = {
                    "finish_reason": finish_reason_of_first_choice,
                    "model_name_returned": getattr(response_data, 'model', self.model_name),
                    "raw_response_id": getattr(response_data, 'id', None),
                }

                all_reasoning_strings = []
                for choice_item in response_data.choices:
                    if hasattr(choice_item, 'message') and choice_item.message:
                        reasoning = getattr(choice_item.message, 'reasoning_content', None)
                        if reasoning:
                            all_reasoning_strings.append(f"<think>\n{str(reasoning)}\n</think>")

                reasoning_prefix = ""
                if all_reasoning_strings:
                    reasoning_prefix = "\n".join(all_reasoning_strings) + "\n\n"

                # --- Determine final output based on tool calls ---
                output_for_generation = ""
                if message_of_first_choice and message_of_first_choice.tool_calls:
                    # If tool calls exist, use the JSON representation of the first choice
                    output_for_generation = first_choice.json()
                elif message_of_first_choice:
                    # Otherwise, use the content of the first choice's message
                    output_for_generation = message_of_first_choice.content or ""

                final_text_output = reasoning_prefix + output_for_generation
                # No longer need to add tool_calls to generation_info separately

                # --- Token Usage Extraction ---
                current_generation_usage = {}
                if hasattr(response_data, 'usage') and response_data.usage:
                    usage_info = response_data.usage
                    for key, value in vars(usage_info).items():
                        if isinstance(value, (int, float)):
                            current_generation_usage[key] = value
                            aggregated_token_usage[key] = aggregated_token_usage.get(key, 0) + value
                        elif value is not None and hasattr(value, '__dict__'):
                            current_generation_usage[key] = {}
                            if key not in aggregated_token_usage:
                                aggregated_token_usage[key] = {}
                            for nested_key, nested_value in vars(value).items():
                                if isinstance(nested_value, (int, float)):
                                    current_generation_usage[key][nested_key] = nested_value
                                    aggregated_token_usage[key][nested_key] = aggregated_token_usage[key].get(
                                        nested_key, 0) + nested_value

                    generation_info["token_usage"] = current_generation_usage
                else:
                    generation_info["warning"] = "Token usage information was not available in the response."

            except Exception as e:
                raise ValueError(f"API call or processing failed for prompt '{prompt_text[:50]}...': {e}") from e

            ai_message = AIMessage(content=final_text_output)
            generation_info["output_token_details"] = {**current_generation_usage}
            if "completion_tokens_details" in current_generation_usage:
                if "reasoning_tokens" in current_generation_usage["completion_tokens_details"]:
                    generation_info["output_token_details"] = {"reasoning" : current_generation_usage["completion_tokens_details"]["reasoning_tokens"]}
            generation_info.update(current_generation_usage)
            generation_info.update({
                "output_tokens": generation_info["completion_tokens"],
                "input_tokens": generation_info["prompt_tokens"],
            })
            ai_message.usage_metadata = generation_info
            chat_generation_obj = ChatGeneration(
                message=ai_message,
                generation_info=generation_info
            )
            generations = [[chat_generation_obj]] + generations

        llm_output_info = {"model_name": self.model_name}
        if aggregated_token_usage:
            llm_output_info["token_usage"] = aggregated_token_usage.copy()
            reasoning_tokens_value = aggregated_token_usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
            if reasoning_tokens_value > 0:
                llm_output_info['reasoning_tokens'] = reasoning_tokens_value

        response = LLMResult(
            generations=generations,
            llm_output=llm_output_info,
        )

        return response