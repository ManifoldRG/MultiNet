from src.modules.source_modules.openai_module import OpenAIModule
from typing import Any, Union


class VLMModule:
    def __init__(self, source: str, model: str) -> None:
        self.source_module = None
        if source == 'openai': 
            self.source_module = OpenAIModule(model)

        assert self.source_module is not None, "The source module has not been set correcly. Check required."
    
    # cur_inputs: (B, N) => Each element is tuple (type, value)
    # Any image inputs should be tagged with the type starting with 'image...' to be correctly converted into image input.
    # Any other data will be considered into text data.
    # k_shots_examples: (B, k, N) => Each element is tuple (input/output, type, value)
    # Each element should indicate if it is input or output when included as a few-shot example.
    # Otherwise, the few-shot prompting would not work as intended.
    def infer_step(self, 
                   cur_inputs: list[list[tuple[str, Any]]], 
                   k_shots_examples: list[list[tuple[str, Any]]]=[],
                   instructions: list[str]=[]
                ) -> list[str]:
        if isinstance(self.source_module, OpenAIModule):
            processed_cur_inputs, processed_k_shots_examples = self._process_inputs_for_api(cur_inputs, k_shots_examples)
            batch_size = len(processed_cur_inputs)

            # The close-sourced API call only support batch size 1.
            outputs = []
            for b in range(batch_size):
                num_examples = 0
                if len(processed_cur_inputs) == len(processed_k_shots_examples):
                    num_examples = len(processed_k_shots_examples[b])
                for k in range(num_examples):
                    for data in processed_k_shots_examples[b][k]:
                        if isinstance(data, list):
                            self.source_module.add_multi_modal_data('input', data)
                        else:
                            self.source_module.add_text_data('output', data)

                output = self.source_module.infer_step_with_images(processed_cur_inputs[b], instructions[b])
                outputs.append(output)

                # Clearing the record.
                self.source_module.clear_history()

            return outputs
        
    # Translating the input data into image/text format.
    def _process_inputs_for_api(self, 
                   cur_inputs: list[list[tuple[str, Any]]], 
                   k_shots_examples: list[list[tuple[str, Any]]]=[]
                ) -> tuple[list[list[dict]], list[list[Union[list[dict], str]]]]:
        batch_size = len(cur_inputs)
        processed_cur_inputs, processed_k_shots_examples = [], []
        for b in range(batch_size):
            processed_inputs = []
            for type, value in cur_inputs[b]:
                if type.startswith('image'):
                    processed_inputs.append({'image': value})
                else:
                    processed_inputs.append({'text': self._convert_into_text(type, value)})
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
                                    data.append({'image': value})
                                else:
                                    data.append({'text': self._convert_into_text(type, value)})
                            processed.append(data)
                        else:
                            raise NotImplementedError("The k-shot examples should have category either 'input' or 'output'.")
                    processed_k_shots_examples[b].append(processed)

        return processed_cur_inputs, processed_k_shots_examples

    # Converting the key-value pair in the input into a text form.
    def _convert_into_text(self, key: str, value: Any) -> str:
        try:
            value_str = str(value)
            return f"{key}: {value_str}"
        except:
            raise ValueError(f"The value {value} cannot be converted into text.")
