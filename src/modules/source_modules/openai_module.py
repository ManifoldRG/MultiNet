from openai import OpenAI
from typing import Any
from PIL import Image
from io import BytesIO

import os
import tiktoken
import math
import numpy as np
import base64
import json
import time
import warnings

CONTEXT_SIZE_MAP = {
    'gpt-4.1-2025-04-14': 1000000,
    'gpt-4o': 128000,
    'gpt-4o-2024-05-13': 128000,
    'gpt-4o-2024-08-06': 128000,
    'chatgpt-4o-latest': 128000,
    'gpt-4o-mini': 128000,
    'gpt-4o-mini-2024-07-18	': 128000,
    'gpt-4-turbo': 128000,
    'gpt-4-turbo-2024-04-09': 128000,
    'gpt-4-turbo-preview': 128000,
    'gpt-4-0125-preview': 128000,
    'gpt-4-1106-preview': 128000,
    'gpt-4': 8196,
    'gpt-4-0613': 8196
}

BATCH_QUEUE_TOKEN_DAY_LIMIT = {
    'gpt-4.1-2025-04-14': 15000000000,
    'gpt-4o': 15000000000,
    'gpt-4o-2024-05-13': 15000000000,
    'gpt-4o-2024-08-06': 15000000000,
    'chatgpt-4o-latest': 15000000000,
    'gpt-4o-mini': 15000000000,
    'gpt-4o-mini-2024-07-18	': 15000000000,
    'gpt-4-turbo': 300000000,
    'gpt-4-turbo-2024-04-09': 300000000,
    'gpt-4-turbo-preview': 300000000,
    'gpt-4-0125-preview': 150000000,
    'gpt-4-1106-preview': 150000000,
    'gpt-4': 150000000,
    'gpt-4-0613': 150000000
}

class OpenAIModule:
    def __init__(self, model: str, max_concurrent_prompts: int = None, max_output_tokens_per_query=256, save_batch_queries=False) -> None:
        if model not in CONTEXT_SIZE_MAP:
            raise KeyError(f"The model {model} is not currenly supported.")
        
        self._max_concurrent_prompts = max_concurrent_prompts if max_concurrent_prompts else 1
        self.history = [[] for _ in range(self._max_concurrent_prompts)]
        self.model = model
        self.max_num_tokens = CONTEXT_SIZE_MAP[model]
        self.cur_num_tokens_cache = [[] for _ in range(self._max_concurrent_prompts)]
        self._batch_job_ids = []
        self.client = OpenAI()
        self.save_batch_queries = save_batch_queries
        
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding('cl100k_base')
        self.batch_queue_token_day_limit = BATCH_QUEUE_TOKEN_DAY_LIMIT[model]
        self.max_output_tokens_per_query = max_output_tokens_per_query
        
    @property
    def batch_job_ids(self):
        return self._batch_job_ids
    
    def clear_batch_job_ids(self):
        self._batch_job_ids = []
    
    @property
    def max_concurrent_prompts(self):
        return self._max_concurrent_prompts
    
    @max_concurrent_prompts.setter
    def max_concurrent_prompts(self, value: int):
        # append empty lists to the history and cur_num_tokens_cache
        if value > len(self.history):
            for _ in range(value - len(self.history)):
                self.history.append([])
                self.cur_num_tokens_cache.append([])
        # remove the last elements from the history and cur_num_tokens_cache
        elif value < len(self.history):
            values_to_remove = len(self.history) - value
            warnings.warn(f'Removing the last {values_to_remove} elements from the history '
                          f'and cur_num_tokens_cache to match max concurrent prompts')
            for _ in range(values_to_remove):
                self.history.pop()
                self.cur_num_tokens_cache.pop()
        
        self._max_concurrent_prompts = value
        
    # One inference step.
    def infer_step(self, inputs: list[tuple[str, Any]], system_prompt: str=None) -> str:
        assert len(inputs) > 0, "The inputs cannot be empty. The inputs should be included."
        self.add_data('input', inputs)
        response = self._get_response_from_api(system_prompt)
        self.add_data('output', [('text', response)])
        return response
    
    def batch_infer_step(self, inputs_batch: list[list[tuple[str, Any]]], 
                         system_prompts: list[str],
                         retrieve_and_return_results: bool = True) -> tuple[list[str], str, int]:
        assert len(inputs_batch) > 0, "There must be at least one input in the batch."
        assert len(inputs_batch) <= len(self.history), "The batch size should be <= the maximum concurrent prompts."
        for inputs in inputs_batch:
            assert len(inputs) > 0, "The inputs cannot be empty. The inputs should be included."
        assert len(inputs_batch) == len(system_prompts), "The number of system prompts should equal batch size."
        
        for i, inputs in enumerate(inputs_batch):
            self.add_data('input', inputs, i)
        input_tokens = sum([sum(num_tokens) for num_tokens in self.cur_num_tokens_cache])
        
        responses, batch_job_id = self._execute_batch_job(system_prompts, 
                                                          retrieve_and_return_results=retrieve_and_return_results)
        
        # adding max out tokens to the count because we may not know the actual number of out tokens
        self.batch_queue_token_day_limit -= (input_tokens + self.max_output_tokens_per_query)
        
        for r, response in enumerate(responses):
            self.add_data('output', [('text', response)], r)
        return responses, batch_job_id, input_tokens
        
    # Adding new data in the context history.
    def add_data(self, type: str, data: list[tuple[str, Any]], idx=0) -> None:
        assert idx < len(self.history), "Index should be less than the maximum concurrent prompts."
        assert type == 'input' or type == 'output', "The data type should be either 'input' or 'output'."
        role = 'user' if type == 'input' else 'assistant'
        message = {'role': role, 'content': []}

        image_sizes = []
        for tup in data:
            if tup[0] == 'text':
                text = tup[1]
                message['content'].append({'type': 'text', 'text': text})
            elif tup[0] == 'image':
                image = tup[1]
                image_url, image_size = self._process_image_for_api(image)
                message['content'].append({'type': 'image_url', 'image_url': {'url': image_url}})
                image_sizes.append(image_size)
            else:
                raise NotImplementedError("OpenAIModule only supports the data type 'text' or 'image'.")
            
        num_tokens = self._get_num_tokens(role, message['content'], image_sizes)
        if self.batch_queue_token_day_limit - num_tokens < 0:
            raise Exception('Exceeded token per day limit') 

        self.history[idx].append(message)
        self.cur_num_tokens_cache[idx].append(num_tokens)
        assert len(self.history[idx]) == len(self.cur_num_tokens_cache[idx]), \
            "The chat history and num tokens cache should be synced."
                
    # Clearing the history.
    def clear_history(self) -> None:
        self.history = [[] for _ in range(self.max_concurrent_prompts)]
        self.cur_num_tokens_cache = [[] for _ in range(self.max_concurrent_prompts)]

    # Calling the chat completion API.
    def _get_response_from_api(self, system_prompt: str=None) -> str:
        start_idx = self._find_starting_point(system_prompt, self.max_output_tokens_per_query)
        system_message = []
        if system_prompt is not None:
            system_message.append({'role': 'system', 'content': system_prompt})

        messages = system_message + self.history[0][start_idx:]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_output_tokens_per_query
        )

        return response.choices[0].message.content
    
    def get_batch_job_status(self, batch_job_id: str) -> bool:
        batch_job = self.client.batches.retrieve(batch_job_id)
        return batch_job.status
    
    def retrieve_batch_results(self, batch_job_id: str) -> list[str]:
        assert self.get_batch_job_status(batch_job_id) == 'completed', \
            "The batch job should be completed to retrieve the results."
        batch_job = self.client.batches.retrieve(batch_job_id)
        result = self.client.files.content(batch_job.output_file_id)
        
        responses = []
        for response in result.iter_lines():
            response = json.loads(response)
            response = response['response']['body']['choices'][0]['message']['content']
            responses.append(response)
        return responses
        
    def _execute_batch_job(self, system_prompts: list[str], 
                           retrieve_and_return_results: bool) -> tuple[list[str], str]:
        start_idxs = []
        for s, system_prompt in enumerate(system_prompts):
            start_idx = self._find_starting_point(system_prompt, self.max_output_tokens_per_query, idx=s)
            start_idxs.append(start_idx)
            
        system_messages = []
        for prompt in system_prompts:
            system_messages.append([{'role': 'system', 'content': prompt}])

        file_name = "batch_queries.jsonl"
        with open(file_name, 'w') as file:
            for i, start_idx in enumerate(start_idxs):
                messages = self.history[i][start_idx:]
                task = {
                    "custom_id": f"task-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "max_tokens": self.max_output_tokens_per_query,
                        "response_format": { 
                            "type": "text"
                        },
                        "messages": system_messages[i] + messages
                    }
                }
        
                file.write(json.dumps(task) + '\n')

        batch_file = self.client.files.create(
            file=open(file_name, "rb"),
            purpose="batch"
        )
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        batch_job_id = batch_job.id
        
        if self.save_batch_queries:
            os.rename(file_name, f"batch_queries_{batch_job.id}.jsonl")
        
        self._batch_job_ids.append(batch_job_id)
        
        if retrieve_and_return_results:
            batch_job = self.client.batches.retrieve(batch_job_id)
            
            sleep_time = 1
            while batch_job.status != 'completed':
                if batch_job.status == 'failed':
                    raise Exception(f"Batch job {batch_job_id} failed.")
                
                # Exponential backoff for polling the batch job status.
                time.sleep(sleep_time)
                sleep_time *= 1.5
                batch_job = self.client.batches.retrieve(batch_job_id)
                
            result = self.client.files.content(batch_job.output_file_id)
            
            responses = []
            for response in result.iter_lines():
                response = json.loads(response)
                response = response['response']['body']['choices'][0]['message']['content']
                responses.append(response)
            return responses, batch_job_id
        else:
            return [], batch_job_id
        
    # Calculating the number of tokens in one message.
    def _get_num_tokens(self, role: str, content: list[dict], image_sizes: list[tuple[int, int]]=[]) -> int:
        num_tokens = 3  # <|start|>, \n, <|end|>
        num_tokens += len(self.encoding.encode(role))
        for obj in content:
            if obj['type'] == 'text':
                num_tokens += len(self.encoding.encode(obj['text']))
        for image_size in image_sizes:
            num_tokens += self._get_num_image_tokens(image_size[0], image_size[1], detail='high')
            
        return num_tokens

    # Utility function for calculating the image tokens.
    # https://community.openai.com/t/how-do-i-calculate-image-tokens-in-gpt4-vision
    def _get_num_image_tokens(self, width: int, height: int, detail: str='high'):
        if detail == 'low':
            return 85
        
        if width > 2048 or height > 2048:
            max_size = 2048
            ratio = width / height
            if ratio > 1.0:
                width = max_size
                height = int(max_size / ratio)
            else:
                height = max_size
                width = int(max_size * ratio)

        min_size = 768
        ratio = width / height
        if width > min_size and height > min_size:
            if ratio > 1.0:
                height = min_size
                width = int(min_size * ratio)
            else:
                width = min_size
                height = int(min_size / ratio)

        num_tiles_width = math.ceil(width / 512)
        num_tiles_height = math.ceil(height / 512)
        return 85 + 170 * (num_tiles_width * num_tiles_height)
    
    # Processing the batched input according to the source module.
    def _process_image_for_api(self, image: np.array) -> tuple[str, tuple]:  # image: (W, H, C)
        image_size = (image.shape[0], image.shape[1])
        image_url = self._convert_image_into_url(image)
        return image_url, image_size

    # Converting the image array into URLs for API calls.
    def _convert_image_into_url(self, image: np.array) -> str:  # image: (W, H, 4)
        image_converted = Image.fromarray(image)
        
        buffer = BytesIO()
        image_converted.save(buffer, format='png')

        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    
    # Finding the starting index of the chat history for adjusting the input size.
    def _find_starting_point(self, system_prompt: str=None, max_response_tokens: int=128, idx=0) -> int:
        num_tokens = 0
        if system_prompt is not None: 
            num_tokens = self._get_num_tokens(role='system', content=[{'type': 'text', 'text': system_prompt}])
            assert num_tokens < self.max_num_tokens, \
                "The number of tokens in the system message must be smaller than the context size."

        start_idx = len(self.cur_num_tokens_cache[idx])
        for i in range(len(self.cur_num_tokens_cache[idx])-1, -1, -1):
            if num_tokens + self.cur_num_tokens_cache[idx][i] > (self.max_num_tokens - max_response_tokens):
                break
            num_tokens += self.cur_num_tokens_cache[idx][i]
            start_idx = i

        return start_idx
