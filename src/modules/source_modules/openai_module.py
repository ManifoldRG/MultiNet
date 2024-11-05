from openai import OpenAI
from typing import Any
from PIL import Image
from io import BytesIO

import tiktoken
import math
import numpy as np
import base64

CONTEXT_SIZE_MAP = {
    'gpt-4o': 128000,
    'gpt-4o-2024-05-13': 128000,
    'gpt-4o-2024-08-06': 128000,
    'chatgpt-4o-latest	': 128000,
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


class OpenAIModule:
    def __init__(self, model: str) -> None:
        if model not in CONTEXT_SIZE_MAP:
            raise KeyError(f"The model {model} is not currenly supported.")
        
        self.history = []
        self.model = model
        self.max_num_tokens = CONTEXT_SIZE_MAP[model]
        self.cur_num_tokens_cache = []

        self.client = OpenAI()
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding('cl100k_base')

    # One inference step.
    def infer_step(self, inputs: list[tuple[str, Any]], system_prompt: str=None) -> str:
        assert len(inputs) > 0, "The inputs cannot be empty. The inputs should be included."
        self.add_data('input', inputs)
        response = self._get_response_from_api(system_prompt)
        self.add_data('output', [('text', response)])
        return response
        
    # Adding new data in the context history.
    def add_data(self, type: str, data: list[tuple[str, Any]]) -> None:
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
        self.history.append(message)
        self.cur_num_tokens_cache.append(self._get_num_tokens(message['role'], message['content'], image_sizes))
                
        assert len(self.history) == len(self.cur_num_tokens_cache), "The chat history and num tokens cache should be synced."

    # Clearing the history.
    def clear_history(self) -> None:
        self.history = []
        self.cur_num_tokens_cache = []

    # Calling the chat completion API.
    def _get_response_from_api(self, system_prompt: str=None) -> str:
        start_idx = self._find_starting_point(system_prompt)
        system_message = []
        if system_prompt is not None:
            system_message.append({'role': 'user', 'content': system_prompt})

        messages = system_message + self.history[start_idx:]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=128
        )

        return response.choices[0].message.content
    
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
    def _find_starting_point(self, system_prompt: str=None, max_response_tokens: int=128) -> int:
        num_tokens = 0
        if system_prompt is not None: 
            num_tokens = self._get_num_tokens(role='system', content=[{'type': 'text', 'text': system_prompt}])
            assert num_tokens < self.max_num_tokens, "The number of tokens in the system message must be smaller than the context size."
        
        start_idx = len(self.cur_num_tokens_cache)
        for i in range(len(self.cur_num_tokens_cache)-1, -1, -1):
            if num_tokens + self.cur_num_tokens_cache[i] > (self.max_num_tokens - max_response_tokens):
                break
            num_tokens += self.cur_num_tokens_cache[i]
            start_idx = i

        return start_idx
