from openai import OpenAI
from typing import Union
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
    def __init__(self, system_prompt: str, model: str) -> None:
        if model not in CONTEXT_SIZE_MAP:
            raise KeyError(f"The model {model} is not currenly supported.")
        
        self.history = []
        self.model = model
        self.max_num_tokens = CONTEXT_SIZE_MAP[model]
        self.cur_num_tokens_cache = []

        self.system_message = {'role': 'system', 'content': system_prompt}
        self.client = OpenAI()
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding('cl100k_base')

    # One inference step.
    def infer_step(self, text: str=None, images: np.array=None) -> str:
        assert text is not None or images is not None, "Either text or images should not be None."
        self.add_data_into_history('input', text, images)
        response = self.get_response_from_api()
        self.add_data_into_history('output', response)
        return response
    
    # Adding new data in the context history.
    def add_data_into_history(self, type: str, text: str=None, images: np.array=None) -> None:  # images: (F, W, H, C) or (F, W, H)
        assert type == 'input' or type == 'output', "The data type should be either 'input' or 'output'."
        assert text is not None or images is not None, "Either text or images should not be None."

        role = 'user' if type == 'input' else 'assistant'
        if images is None:
            message = {'role': role, 'content': text}
            self.history.append(message)
            self.cur_num_tokens_cache.append(self.get_text_message_num_tokens(message['role'], message['content']))
        else:
            image_urls, image_size = self.process_images_for_api(images)
            message = {'role': role, 'content': []}
            for image_url in image_urls:
                message['content'].append({'type': 'image_url', 'image_url': {'url': image_url}})
            if text is not None: message['content'].append({'type': 'text', 'text': text})
            self.history.append(message)
            self.cur_num_tokens_cache.append(self.get_multi_modal_message_num_tokens(message['role'], [image_size] * images.shape[0], text))

        assert len(self.history) == len(self.cur_num_tokens_cache), "The chat history and num tokens cache should be synced."

    # Clearing the history.
    def clear_history(self):
        self.history = []
        self.cur_num_tokens_cache = []

    # Calling the chat completion API.
    def get_response_from_api(self) -> str:
        start_idx = self.find_starting_point()
        messages = [self.system_message] + self.history[start_idx:]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=128
        )

        return response.choices[0].message.content

    # Calculating the number of tokens in one message only with text.
    def get_text_message_num_tokens(self, role: str, content: str) -> int:
        num_tokens = 3  # <|start|>, \n, <|end|>
        num_tokens += len(self.encoding.encode(role))
        num_tokens += len(self.encoding.encode(content))
        return num_tokens
    
    # Calculating the number of tokens in one message.
    def get_multi_modal_message_num_tokens(self, role: str, image_sizes: list[tuple[int, int]], text: str=None) -> int:
        num_tokens = 3
        num_tokens += len(self.encoding.encode(role))
        for image_size in image_sizes:
            num_tokens += self.get_num_image_tokens(image_size[0], image_size[1], detail='high')
        if text is not None: num_tokens += len(self.encoding.encode(text))
            
        return num_tokens

    # Utility function for calculating the image tokens.
    # https://community.openai.com/t/how-do-i-calculate-image-tokens-in-gpt4-vision
    def get_num_image_tokens(self, width: int, height: int, detail: str='high'):
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
    def process_images_for_api(self, images: np.array) -> tuple[list, tuple]:  # images: (F, W, H, C)
        image_size = (images.shape[1], images.shape[2])
        image_urls = []
        for image in images:
            image_url = self.convert_image_into_url(image)
            image_urls.append(image_url)
        return image_urls, image_size

    # Converting the image array into URLs for API calls.
    def convert_image_into_url(self, image: np.array) -> str:
        # Checking the data spec.
        multiplier, mode = 1.0, 'RGB'
        if len(image.shape) == 2:  # Grey-scaled image.
            mode = 'L'
        if len(image.shape) == 3 and image.shape[-1] == 1:  # Grey-scaled image with extra dimension.
            mode = 'L'
            image = np.squeeze(image, axis=-1)
        if np.max(image) <= 1.0:  # Setting the values in range of 0 ~ 255.
            multiplier = 255
        image_converted = Image.fromarray((image * multiplier).astype(np.uint8), mode)
        
        buffer = BytesIO()
        image_converted.save(buffer, format='png')

        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    
    # Finding the starting index of the chat history for adjusting the input size.
    def find_starting_point(self, max_response_tokens: int=128) -> int:
        num_tokens = self.get_text_message_num_tokens(self.system_message)
        assert num_tokens < self.max_num_tokens, "The number of tokens in the system message must be smaller than the context size."
        
        start_idx = len(self.cur_num_tokens_cache)
        for i in range(len(self.cur_num_tokens_cache)-1, -1, -1):
            if num_tokens + self.cur_num_tokens_cache[i] > (self.max_num_tokens - max_response_tokens):
                break
            num_tokens += self.cur_num_tokens_cache[i]
            start_idx = i

        return start_idx
