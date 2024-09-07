import os
import sys
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
os.environ["OPENAI_API_KEY"] = "random-api-key"

from src.modules.source_modules.openai_module import OpenAIModule
from unittest.mock import MagicMock

import unittest


class OpenAIModuleTest(unittest.TestCase):
    def test_constructor(self):
        system_prompt = "This is a test system prompt1."
        model = "random-model"
        with self.assertRaises(KeyError):
            module = OpenAIModule(system_prompt, model)

        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(system_prompt, model)
        self.assertEqual(module.model, model)
        self.assertEqual(module.system_message, {'role': 'system', 'content': system_prompt})
        self.assertEqual(module.max_num_tokens, 128000)
        self.assertEqual(module.cur_num_tokens_cache, [])
        self.assertEqual(module.encoding.name, 'o200k_base')

        system_prompt = "This is a test system prompt2."
        model = "gpt-4"
        module = OpenAIModule(system_prompt, model)
        self.assertEqual(module.model, model)
        self.assertEqual(module.system_message, {'role': 'system', 'content': system_prompt})
        self.assertEqual(module.max_num_tokens, 8196)
        self.assertEqual(module.cur_num_tokens_cache, [])
        self.assertEqual(module.encoding.name, 'cl100k_base')

    def test_chat_completion_basic(self):
        system_prompt = "This is a test system prompt."
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(system_prompt, model)

        queries = [
            "What's your name?",
            "What is the advantages of Python language?",
            "What is the definition of a generalist foundation model?"
        ]
        for q, query in enumerate(queries):
            module.get_response_from_api = MagicMock(return_value=f"This is a response {q}.")
            response = module.chat_completion_basic(query)
            self.assertEqual(response, f"This is a response {q}.")
            self.assertEqual(module.messages[q*2], {'role': 'user', 'content': query})
            self.assertEqual(module.messages[q*2+1], {'role': 'assistant', 'content': response})
            self.assertEqual(module.cur_num_tokens_cache[q*2], module.get_message_num_tokens({'role': 'user', 'content': query}))
            self.assertEqual(module.cur_num_tokens_cache[q*2+1], module.get_message_num_tokens({'role': 'assistant', 'content': response}))
            self.assertEqual(len(module.messages), q*2+2)
            self.assertEqual(len(module.cur_num_tokens_cache), q*2+2)

    def test_chat_completion_multi_modal(self):
        system_prompt = "This is a test system prompt."
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(system_prompt, model)

        queries = [
            {
                'image_urls': ['image_url_0_0', 'image_url_0_1', 'image_url_0_2'], 
                'image_sizes': [(128, 128), (512, 128), (1200, 200)],
                'text_query': "What are these pictures showing?"
            },
            {
                'image_urls': ['image_url_1_0', 'image_url_1_1', 'image_url_1_2', 'image_url_1_3', 'image_url_1_4'],
                'image_sizes': [(640, 640), (100, 100), (136, 780), (100, 100), (100, 100)],
                'text_query': "Find a picture of a dog from these pictures."
            },
            {
                'image_urls': ['image_url_2_0'],
                'image_sizes': [(1920, 1080)]
            },
            {
                'image_urls': ['image_url_3_0', 'image_url_3_1'],
                'image_sizes': [(720, 720), (1368, 960)]
            }
        ]
        for q, query in enumerate(queries):
            module.get_response_from_api = MagicMock(return_value=f"This is a response {q}.")
            image_urls, image_sizes, text_query = query['image_urls'], query['image_sizes'],  query['text_query'] if 'text_query' in query else None
            expected_content = [{'type': 'image_url', 'image_url': {'url': image_url}} for image_url in query['image_urls']]
            if text_query is not None: expected_content.append({'type': 'text', 'text': query['text_query']})

            response = module.chat_completion_multi_modal(image_urls, image_sizes, text_query)
            self.assertEqual(response, f"This is a response {q}.")
            self.assertEqual(module.messages[q*2], {'role': 'user', 'content': expected_content})
            self.assertEqual(module.messages[q*2+1], {'role': 'assistant', 'content': response})

            content = []
            for image_url in image_urls:
                content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': image_url
                    }
                })
            if text_query is not None: content.append({'type': 'text', 'text': text_query})
            self.assertEqual(module.cur_num_tokens_cache[q*2], module.get_message_num_tokens({'role': 'user', 'content': content}, image_sizes))
            self.assertEqual(module.cur_num_tokens_cache[q*2+1], module.get_message_num_tokens({'role': 'assistant', 'content': response}))
            self.assertEqual(len(module.messages), q*2+2)
            self.assertEqual(len(module.cur_num_tokens_cache), q*2+2)

    def test_get_num_image_tokens(self):
        system_prompt = "This is a test system prompt."
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(system_prompt, model)

        self.assertEqual(module.get_num_image_tokens(128, 1024, 'low'), 85)
        self.assertEqual(module.get_num_image_tokens(4096, 4096, 'low'), 85)
        self.assertEqual(module.get_num_image_tokens(512, 64, 'low'), 85)
        self.assertEqual(module.get_num_image_tokens(128, 128), 255)
        self.assertEqual(module.get_num_image_tokens(512, 512), 255)
        self.assertEqual(module.get_num_image_tokens(600, 150), 425)
        self.assertEqual(module.get_num_image_tokens(130, 515), 425)
        self.assertEqual(module.get_num_image_tokens(900, 1000), 765)
        self.assertEqual(module.get_num_image_tokens(1280, 1080), 765)
        self.assertEqual(module.get_num_image_tokens(4096, 1920), 1445)
        self.assertEqual(module.get_num_image_tokens(1080, 4096), 1445)

    def test_get_message_num_tokens(self):
        system_prompt = "This is a test system prompt."
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(system_prompt, model)

        def side_effect_encode(text):
            if text == 'system' or text == 'user' or text == 'assistant': return [1]
            elif len(text) < 10: return [1,2,3,4,5]
            elif len(text) < 20: return [1,2,3,4,5,6,7,8,9,10]
            elif len(text) < 30: return [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            else: return [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        def side_effect_image(width, height, detail):
            if detail == 'low': return 8
            else:
                if width == 128 and height == 128: return 24
                elif width == 512 and height == 512: return 64
                elif width == 1024 and height == 1024: return 128
                else: return 55
        module.encoding.encode = MagicMock(side_effect=side_effect_encode)
        module.get_num_image_tokens = MagicMock(side_effect=side_effect_image)

        self.assertEqual(module.get_message_num_tokens(module.system_message), 20)
        self.assertEqual(module.get_message_num_tokens({'role': 'user', 'content': "What's your name?"}), 14)
        self.assertEqual(module.get_message_num_tokens({'role': 'assistant', 'content': "Live as if you were to die tomorrow."}), 28)

        self.assertEqual(module.get_message_num_tokens({
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': "random-image1.jpg"}},
                {'type': 'image_url', 'image_url': {'url': "random-image2.jpg"}},
                {'type': 'image_url', 'image_url': {'url': "random-image3.jpg"}}
            ]
        }, [(128, 128), (80, 190), (512, 512)]), 147)
        self.assertEqual(module.get_message_num_tokens({
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': "random-image1.jpg"}},
                {'type': 'text', 'text': "Good job."}
            ]
        }, [(1024, 1024)]), 137)
        self.assertEqual(module.get_message_num_tokens({
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': "random-image1.jpg"}},
                {'type': 'image_url', 'image_url': {'url': "random-image2.jpg"}},
                {'type': 'image_url', 'image_url': {'url': "random-image3.jpg"}},
                {'type': 'image_url', 'image_url': {'url': "random-image4.jpg"}},
                {'type': 'text', 'text': "What is your name?"}
            ]
        }, [(64, 64), (1024, 1024), (640, 512), (128, 120)]), 307)
        

    def test_find_starting_point(self):
        system_prompt = "This is a test system prompt."
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(system_prompt, model)
        module.max_num_tokens = 1152

        module.get_message_num_tokens = MagicMock(return_value=10)
        module.cur_num_tokens_cache = [100, 100, 100, 100, 100]
        self.assertEqual(module.find_starting_point(), 0)
        module.cur_num_tokens_cache = [500, 200, 300, 400]
        self.assertEqual(module.find_starting_point(), 1)
        module.cur_num_tokens_cache = [57, 100, 231, 10, 490, 126]
        self.assertEqual(module.find_starting_point(), 0)
        module.cur_num_tokens_cache = [68, 367, 111, 77, 132, 2, 100, 501, 92]
        self.assertEqual(module.find_starting_point(), 3)

        module.max_num_tokens = 11
        module.cur_num_tokens_cache = [10, 5, 18, 24]
        self.assertEqual(module.find_starting_point(), 4)


if __name__=="__main__":
    unittest.main()
