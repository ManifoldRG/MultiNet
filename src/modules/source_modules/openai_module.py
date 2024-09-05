from openai import OpenAI
from typing import Union

import tiktoken
import math


class OpenAIModule:
    def __init__(self, system_prompt: str, model: str, max_num_tokens: int=4096) -> None:
        self.messages = []
        self.model = model
        self.max_num_tokens = max_num_tokens
        self.cur_num_tokens_cache = []
        self.system_message = {'role': 'system', 'content': system_prompt}

        self.client = OpenAI()
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding('cl100k_base')

    # One chat completion only with text messages.
    def chat_completion_basic(self, text_query: str) -> str:
        self.add_message({'role': 'user', 'content': text_query})
        response = self.get_response_from_api()
        self.add_message({'role': 'assistant', 'content': response})
        return response

    # One chat completion with text + images.
    def chat_completion_multi_modal(self, image_urls: list[str], image_sizes: list[tuple[int, int]], text_query: str=None) -> str:
        content = []
        for image_url in image_urls:
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': image_url
                }
            })
        if text_query is not None: content.append({'type': 'text', 'text': text_query})
        self.add_message({'role': 'user', 'content': content}, image_sizes)
        response = self.get_response_from_api()
        self.add_message({'role': 'assistant', 'content': response})
        return response

    # Calling the chat completion API.
    def get_response_from_api(self) -> str:
        start_idx = self.find_starting_point()
        messages = [self.system_message] + self.messages[start_idx:]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return response.choices[0].message.content
    
    # Adding a message into the chat history.
    def add_message(self, message: dict[str, Union[str, list[dict]]], image_sizes: list[tuple[int, int]]=None):
        num_tokens = self.get_message_num_tokens(message, image_sizes)
        self.messages.append(message)
        self.cur_num_tokens_cache.append(num_tokens)
        assert len(self.messages) == len(self.cur_num_tokens_cache), "The chat history and num tokens cache should be synced."
    
    # Calculating the number of tokens in one message only with text.
    def get_text_message_num_tokens(self, message: dict[str, str]) -> int:
        num_tokens = 3  # <|start|>, \n, <|end|>
        num_tokens += len(self.encoding.encode(message['role']))
        num_tokens += len(self.encoding.encode(message['content']))
        return num_tokens
    
    # Calculating the number of tokens in one message.
    def get_message_num_tokens(self, message: dict[str, Union[str, list[dict]]], image_sizes: list[tuple[int, int]]=None) -> int:
        if image_sizes is None:
            return self.get_text_message_num_tokens(message)
        
        image_contents, text_content = None, None
        if message['content'][-1]['type'] == 'text':
            image_contents = message['content'][:-1]
            text_content = message['content'][-1]
        else:
            image_contents = message['content']
        
        assert len(image_contents) == len(image_sizes), "The image sizes list should be the same length as the image input list."

        num_tokens = 3
        num_tokens += len(self.encoding.encode(message['role']))
        for i, image_content in enumerate(image_contents):
            num_tokens += self.get_num_image_tokens(image_sizes[i][0], image_sizes[i][1], detail='high')
        if text_content is not None: num_tokens += len(self.encoding.encode(text_content['text']))
            
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
    
    # Finding the starting index of the chat history for adjusting the input size.
    def find_starting_point(self) -> int:
        num_tokens = self.get_message_num_tokens(self.system_message)
        assert num_tokens < self.max_num_tokens, "The number of tokens in the system message must be smaller than the context size."
        
        start_idx = len(self.cur_num_tokens_cache)
        for i in range(len(self.cur_num_tokens_cache)-1, -1, -1):
            if num_tokens + self.cur_num_tokens_cache[i] > self.max_num_tokens:
                break
            num_tokens += self.cur_num_tokens_cache[i]
            start_idx = i

        return start_idx
