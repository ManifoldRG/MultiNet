from src.modules.source_modules.openai_module import OpenAIModule
from typing import Any, Union

import ast


class VLMModule:
    def __init__(self, source: str, model: str, max_concurrent_prompts: int = None, max_output_tokens_per_query: int = None) -> None:
        self.source_module = None
        if source == 'openai': 
            self.source_module = OpenAIModule(model, max_concurrent_prompts, max_output_tokens_per_query)

        assert self.source_module is not None, "The source module has not been set correcly. Check required."
    
    # cur_inputs: (B, N) => Each element is tuple (type, value)
    # k_shots_examples: (B, k, N) => Each element is tuple (input/output, type, value)
    # Each element should indicate if it is input or output when included as a few-shot example.
    # Otherwise, the few-shot prompting would not work as intended.
    def infer_step(self, 
                   cur_inputs: list[list[tuple[str, Any]]], 
                   k_shots_examples: list[list[tuple[str, list[tuple[str, Any]]]]]=[],
                   instructions: list[str]=[],
                   output_types: list[type]=[]
                   ) -> list[str]:
        processed_cur_inputs, processed_k_shots_examples = self._process_inputs(cur_inputs, k_shots_examples)

        outputs = []
        if isinstance(self.source_module, OpenAIModule):
            batch_size = len(processed_cur_inputs)
            for b in range(batch_size):
                num_examples = 0
                if len(processed_cur_inputs) == len(processed_k_shots_examples):
                    num_examples = len(processed_k_shots_examples[b])
                for k in range(num_examples):
                    for data in processed_k_shots_examples[b][k]:
                        if isinstance(data, list):
                            self.source_module.add_data('input', data)
                        else:
                            self.source_module.add_data('output', [('text', data)])

                output = self.source_module.infer_step(processed_cur_inputs[b], instructions[b])
                output_type = str if len(output_types) == 0 else output_types[b]
                outputs.append(self.convert_into_data(output, output_type))

                # Clearing the record.
                self.source_module.clear_history()

        return outputs
    
    def send_batch_job(self, 
                        cur_inputs: list[list[tuple[str, Any]]],
                        k_shots_examples: list[list[tuple[str, list[tuple[str, Any]]]]]=None,
                        instructions: Union[str, list[str]]=None
                        ) -> str:
        if k_shots_examples is None:
            k_shots_examples = []
        if instructions is None:
            instructions = []
        # TODO: Add support for k-shot
        processed_cur_inputs, processed_k_shots_examples = self._process_inputs(cur_inputs, [])

        if isinstance(self.source_module, OpenAIModule):
            if self.source_module.max_concurrent_prompts < len(cur_inputs):
                self.source_module.max_concurrent_prompts = len(cur_inputs)
            _, batch_job_id, num_tokens = self.source_module.batch_infer_step(processed_cur_inputs, instructions, 
                                                                              retrieve_and_return_results=False)
            self.source_module.clear_history()
        return batch_job_id, num_tokens
    
    
    def get_batch_job_status(self, batch_job_id: str) -> str:
        return self.source_module.get_batch_job_status(batch_job_id)
      
    def retrieve_batch_results(self, 
                               batch_job_id: str, 
                               output_types: Union[type, list[type]]=None
                               ) -> list[str]:
        if isinstance(self.source_module, OpenAIModule):
            outputs = self.source_module.retrieve_batch_results(batch_job_id)
            output_type = str
            if output_types is None:
                output_types = []
            if isinstance(output_types, type):
                output_type = output_types
                output_types = []
                
            output_types = [output_type]*len(outputs) if len(output_types) == 0 else output_types
            assert len(outputs) == len(output_types), "The number of outputs should be the same as the number of output types."
            
            outputs = [self.convert_into_data(output, output_type) for output, output_type in zip(outputs, output_types)]          

            return outputs
    
    # Translating the input data into image/text format.
    # Any image inputs should be tagged with the type starting with 'image...' to be correctly converted into image input.
    # Any other data will be considered into text data.
    def _process_inputs(self, 
                        cur_inputs: list[list[tuple[str, Any]]], 
                        k_shots_examples: list[list[tuple[str, list[tuple[str, Any]]]]]=[]
                        ) -> tuple[list[list[dict]], list[list[Union[list[dict], str]]]]:
        batch_size = len(cur_inputs)
        processed_cur_inputs, processed_k_shots_examples = [], []
        for b in range(batch_size):
            processed_inputs = []
            for type, value in cur_inputs[b]:
                if type.startswith('image'):
                    processed_inputs.append(('image', value))
                else:
                    processed_inputs.append(('text', self._convert_into_text(type, value)))
            processed_cur_inputs.append(processed_inputs)

        # If there are k-shots examples, processing them.
        if len(k_shots_examples) > 0:
            assert len(k_shots_examples) == batch_size, "The size of k_shots_examples should be the same as batch size."
            for b in range(batch_size):
                num_examples = len(k_shots_examples[b])
                processed_k_shots_examples.append([])

                for k in range(num_examples):
                    processed = []
                    for tup in k_shots_examples[b][k]:
                        if tup[0] == 'output':
                            processed.append(tup[1])
                        elif tup[0] == 'input':
                            data = []
                            for type, value in tup[1]:
                                if type.startswith('image'):
                                    data.append(('image', value))
                                else:
                                    data.append(('text', self._convert_into_text(type, value)))
                            processed.append(data)
                        else:
                            raise NotImplementedError("The k-shot examples should have category either 'input' or 'output'.")
                    processed_k_shots_examples[b].append(processed)

        return processed_cur_inputs, processed_k_shots_examples

    # Converting the key-value pair in the input into a text form.
    def _convert_into_text(self, key: str, value: Any) -> str:
        if isinstance(value, str):
            # For strings, use the value directly to preserve newlines and special characters
            value_str = value
        else:
            # For non-string types, convert to string
            value_str = str(value)
        return f"{key}: {value_str}"

    # Converting the text output into the requested form of data type.
    def convert_into_data(self, text: str, data_type: type) -> Any:
        try:
            if data_type == str:
                return text
            elif data_type == int:
                return int(text)
            elif data_type == float:
                return float(text)
            elif data_type == list:
                return ast.literal_eval(text)
            elif data_type == tuple:
                first, second = text.split(' ')
                return (first, second)
            else:
                raise NotImplementedError(f"The data type {data_type} is not currenly supported for VLM output.")
        except:
            print(f"Cannot convert {text} into data type '{data_type}'.")
            return None
