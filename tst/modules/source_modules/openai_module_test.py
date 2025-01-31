import os
import sys
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
os.environ["OPENAI_API_KEY"] = "random-api-key"

from src.modules.source_modules.openai_module import OpenAIModule
from unittest.mock import MagicMock

import unittest
import numpy as np


class OpenAIModuleTest(unittest.TestCase):
    def test_constructor(self):
        model = "random-model"
        with self.assertRaises(KeyError):
            module = OpenAIModule(model)

        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(model)
        self.assertEqual(module.model, model)
        self.assertEqual(module.max_num_tokens, 128000)
        self.assertEqual(module.history, [[]])
        self.assertEqual(module.cur_num_tokens_cache, [[]])
        self.assertEqual(module.encoding.name, 'o200k_base')

        model = "gpt-4"
        module = OpenAIModule(model)
        self.assertEqual(module.model, model)
        self.assertEqual(module.max_num_tokens, 8196)
        self.assertEqual(module.history, [[]])
        self.assertEqual(module.cur_num_tokens_cache, [[]])
        self.assertEqual(module.encoding.name, 'cl100k_base')
        
    def test_infer_step_with_texts(self):
        system_prompt = "This is a test system prompt."
        model = "gpt-4o-2024-05-13"

        module = OpenAIModule(model)
        queries = [
            "What's your name?",
            "What is the advantages of Python language?",
            "What is the definition of a generalist foundation model?"
        ]
        for q, query in enumerate(queries):
            module._get_response_from_api = MagicMock(return_value=f"This is a response {q} for single query testing.")
            response = module.infer_step([('text', query)], system_prompt)
            self.assertEqual(response, f"This is a response {q} for single query testing.")
            self.assertEqual(module.history[0][q*2], {'role': 'user', 'content': [{'type': 'text', 'text': queries[q]}]})
            self.assertEqual(module.history[0][q*2+1], {'role': 'assistant', 'content': [{'type': 'text', 'text': response}]})
            self.assertEqual(module.cur_num_tokens_cache[0][q*2], module._get_num_tokens('user', [{'type': 'text', 'text': query}]))
            self.assertEqual(module.cur_num_tokens_cache[0][q*2+1], module._get_num_tokens('assistant', [{'type': 'text', 'text': response}]))
            self.assertEqual(len(module.history[0]), q*2+2)
            self.assertEqual(len(module.cur_num_tokens_cache[0]), q*2+2)

        module = OpenAIModule(model)
        multi_queries = [
            ["This is a user message 1", "Additional user message."],
            ["What's your name?", "My name is Python.", "Nice to meet you."]
        ]
        for m, multi_query in enumerate(multi_queries):
            module._get_response_from_api = MagicMock(return_value=f"This is a response {m} for multi query testing.")
            response = module.infer_step([('text', query) for query in multi_query], system_prompt)
            self.assertEqual(response, f"This is a response {m} for multi query testing.")
            expected_content = [{'type': 'text', 'text': query} for query in multi_query]
            self.assertEqual(module.history[0][m*2], {'role': 'user', 'content': expected_content})
            self.assertEqual(module.history[0][m*2+1], {'role': 'assistant', 'content': [{'type': 'text', 'text': response}]})
            self.assertEqual(module.cur_num_tokens_cache[0][m*2], module._get_num_tokens('user', expected_content))
            self.assertEqual(module.cur_num_tokens_cache[0][m*2+1], module._get_num_tokens('assistant', [{'type': 'text', 'text': response}]))
            self.assertEqual(len(module.history[0]), m*2+2)
            self.assertEqual(len(module.cur_num_tokens_cache[0]), m*2+2)

    def test_batch_infer_with_texts(self):
        model = "gpt-4o-2024-05-13"
        
        # Single object queries
        module = OpenAIModule(model, max_concurrent_prompts=3)
        system_prompts = [
            'This is a test system prompt 1.', 
            'This is a test system prompt 2.', 
            'This is a test system prompt 3.'
        ]
        
        queries = [
            "What's your name?",
            "What is the advantages of Python language?",
            "What is the definition of a generalist foundation model?"
        ]

        mock_responses, model_inputs = [], []
        for q, query in enumerate(queries):
            mock_responses.append(f"This is a response {q} for single query testing.")
            
            model_inputs.append([('text', query)])
        
        module._execute_batch_job = MagicMock(return_value=(mock_responses[:], 'batchid123'))
        responses, batch_job_id, input_tokens = module.batch_infer_step(model_inputs, system_prompts, retrieve_and_return_results=True)
        self.assertEqual(batch_job_id, 'batchid123')
        self.assertEqual(responses, mock_responses)
        total_input_tokens = sum([module._get_num_tokens('user', [{'type': 'text', 'text': query}]) for query in queries])
        self.assertEqual(input_tokens, total_input_tokens)
        
        for q, query in enumerate(queries):
            self.assertEqual(responses[q], f"This is a response {q} for single query testing.")
            self.assertEqual(module.history[q][0], {'role': 'user', 'content': [{'type': 'text', 'text': query}]})
            self.assertEqual(module.history[q][1], {'role': 'assistant', 'content': [{'type': 'text', 'text': responses[q]}]})
            self.assertEqual(module.cur_num_tokens_cache[q][0], module._get_num_tokens('user', [{'type': 'text', 'text': query}]))
            self.assertEqual(module.cur_num_tokens_cache[q][1], module._get_num_tokens('assistant', [{'type': 'text', 'text': responses[q]}]))
            self.assertEqual(len(module.history[q]), 2)
            self.assertEqual(len(module.cur_num_tokens_cache[q]), 2)
            
        # Multi-object queries
        model = "gpt-4o-2024-05-13"
        
        module = OpenAIModule(model, max_concurrent_prompts=3)
        system_prompts = [
            'This is a test system prompt 1.', 
            'This is a test system prompt 2.', 
            'This is a test system prompt 3.'
        ]
        
        multi_queries = [
            ["What is the definition of a generalist foundation model?"],
            ["This is a user message 1", "Additional user message."],
            ["What's your name?", "My name is Python.", "Nice to meet you."]
        ]
        
        mock_responses, model_inputs = [], []
        for m, multi_query in enumerate(multi_queries):
            mock_response = f"This is a response {m} for multi query testing."
            mock_responses.append(mock_response)
            
            model_inputs.append([('text', obj) for obj in multi_query])
        
        module._execute_batch_job = MagicMock(return_value=(mock_responses[:], 'batchid123'))
        responses, batch_job_id, input_tokens = module.batch_infer_step(model_inputs, system_prompts, retrieve_and_return_results=True)
        self.assertEqual(batch_job_id, 'batchid123')
        self.assertEqual(responses, mock_responses)
        total_input_tokens = 0
        for multi_query in multi_queries:
            total_input_tokens += module._get_num_tokens('user', [{'type': 'text', 'text': obj} for obj in multi_query])
        self.assertEqual(input_tokens, total_input_tokens)
        
        for m, multi_query in enumerate(multi_queries):
            self.assertEqual(responses[m], f"This is a response {m} for multi query testing.")
            expected_content = [{'type': 'text', 'text': query} for query in multi_query]
            self.assertEqual(module.history[m][0], {'role': 'user', 'content': expected_content})
            self.assertEqual(module.history[m][1], {'role': 'assistant', 'content': [{'type': 'text', 'text': responses[m]}]})
            self.assertEqual(module.cur_num_tokens_cache[m][0], module._get_num_tokens('user', expected_content))
            self.assertEqual(module.cur_num_tokens_cache[m][1], module._get_num_tokens('assistant', [{'type': 'text', 'text': responses[m]}]))
            self.assertEqual(len(module.history[m]), 2)
            self.assertEqual(len(module.cur_num_tokens_cache[m]), 2)
            
    def test_infer_step_with_images(self):
        system_prompt = "This is a test system prompt."
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(model)

        queries = [
            [
                ('image', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('text', "What are these pictures showing?")
            ],
            [
                ('image', np.random.randint(256, size=(512, 512, 2)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(100, 200, 4)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(256, 128, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(1024, 120, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(180, 260, 3)).astype(np.uint8)),
                ('text', "Find a picture of a dog from these pictures.")
            ],
            [
                ('image', np.random.randint(256, size=(1920, 1080, 3)).astype(np.uint8)),
                ('text', "This is an example figure."),
                ('image', np.random.randint(256, size=(300, 500, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(256, 256, 2)).astype(np.uint8)),
                ('text', "What do you think about other two pictures?")
            ],
            [
                ('image', np.random.randint(256, size=(1000, 1000, 4)).astype(np.uint8))
            ],
            [
                ('image', np.random.randint(256, size=(512, 512, 4)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(128, 64, 3)).astype(np.uint8))
            ]
        ]
        
        for q, query in enumerate(queries):
            module._get_response_from_api = MagicMock(return_value=f"This is a response {q}.")
            response = module.infer_step(query, system_prompt)
            self.assertEqual(response, f"This is a response {q}.")
            self.assertEqual(module.history[0][q*2+1], {'role': 'assistant', 'content': [{'type': 'text', 'text': response}]})
            self.assertEqual(module.history[0][q*2]['role'], 'user')
            self.assertTrue(isinstance(module.history[0][q*2]['content'], list))

            for o, obj in enumerate(query):
                if obj[0] == 'text':
                    self.assertEqual(module.history[0][q*2]['content'][o], {'type': 'text', 'text': obj[1]})
                elif obj[0] == 'image':
                    self.assertEqual(module.history[0][q*2]['content'][o]['type'], 'image_url')
                    self.assertTrue(isinstance(module.history[0][q*2]['content'][o]['image_url']['url'], str))
                    self.assertTrue(module.history[0][q*2]['content'][o]['image_url']['url'].startswith('data:image/png;base64,'))

            self.assertEqual(module.cur_num_tokens_cache[0][q*2], module._get_num_tokens(
                'user',
                 module.history[0][q*2]['content'], 
                [(obj[1].shape[0], obj[1].shape[1]) for obj in query if obj[0] == 'image']
            ))
            self.assertEqual(module.cur_num_tokens_cache[0][q*2+1], module._get_num_tokens('assistant', [{'type': 'text', 'text': response}]))

            self.assertEqual(len(module.history[0]), q*2+2)
            self.assertEqual(len(module.cur_num_tokens_cache[0]), q*2+2)

    def test_batch_infer_with_images(self):
        model = "gpt-4o-2024-05-13"

        queries = [
            [
                ('image', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('text', "What are these pictures showing?")
            ],
            [
                ('image', np.random.randint(256, size=(512, 512, 2)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(100, 200, 4)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(256, 128, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(1024, 120, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(180, 260, 3)).astype(np.uint8)),
                ('text', "Find a picture of a dog from these pictures.")
            ],
            [
                ('image', np.random.randint(256, size=(1920, 1080, 3)).astype(np.uint8)),
                ('text', "This is an example figure."),
                ('image', np.random.randint(256, size=(300, 500, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(256, 256, 2)).astype(np.uint8)),
                ('text', "What do you think about other two pictures?")
            ],
            [
                ('image', np.random.randint(256, size=(1000, 1000, 4)).astype(np.uint8))
            ],
            [
                ('image', np.random.randint(256, size=(512, 512, 4)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(128, 64, 3)).astype(np.uint8))
            ]
        ]
        system_prompts = ["This is a test system prompt." for _ in range(len(queries))]
        module = OpenAIModule(model, max_concurrent_prompts=len(queries))
        mock_responses, model_inputs = [], []
        for q, query in enumerate(queries):
            mock_response = f"This is a response {q}."
            mock_responses.append(mock_response)
            
            model_inputs.append(query)
            
        module._execute_batch_job = MagicMock(return_value=(mock_responses[:], 'batchid123'))
        responses, batch_job_id, input_tokens = module.batch_infer_step(model_inputs, system_prompts, retrieve_and_return_results=True)
        self.assertEqual(batch_job_id, 'batchid123')
        self.assertEqual(responses, mock_responses)
        total_input_tokens = 0
        for q, query in enumerate(queries):
            total_input_tokens += module._get_num_tokens(
                'user',
                 module.history[q][0]['content'], 
                [(obj[1].shape[0], obj[1].shape[1]) for obj in query if obj[0] == 'image']
            )
        self.assertEqual(input_tokens, total_input_tokens)

        for q, query in enumerate(queries):
            self.assertEqual(responses[q], f"This is a response {q}.")
            self.assertEqual(module.history[q][1], {'role': 'assistant', 'content': [{'type': 'text', 'text': responses[q]}]})
            self.assertEqual(module.history[q][0]['role'], 'user')
            self.assertTrue(isinstance(module.history[q][0]['content'], list))

            for o, obj in enumerate(query):
                if obj[0] == 'text':
                    self.assertEqual(module.history[q][0]['content'][o], {'type': 'text', 'text': obj[1]})
                elif obj[0] == 'image':
                    self.assertEqual(module.history[q][0]['content'][o]['type'], 'image_url')
                    self.assertTrue(isinstance(module.history[q][0]['content'][o]['image_url']['url'], str))
                    self.assertTrue(module.history[q][0]['content'][o]['image_url']['url'].startswith('data:image/png;base64,'))

            self.assertEqual(module.cur_num_tokens_cache[q][0], module._get_num_tokens(
                'user',
                 module.history[q][0]['content'], 
                [(obj[1].shape[0], obj[1].shape[1]) for obj in query if obj[0] == 'image']
            ))
            self.assertEqual(module.cur_num_tokens_cache[q][1], module._get_num_tokens('assistant', [{'type': 'text', 'text': responses[q]}]))

            self.assertEqual(len(module.history[q]), 2)
            self.assertEqual(len(module.cur_num_tokens_cache[q]), 2)

    def test_invalid_inputs(self):
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(model)

        with self.assertRaises(AssertionError):
            response = module.infer_step([], None)

        with self.assertRaises(AssertionError):
            module.add_data('dummy', data=[])

        with self.assertRaises(NotImplementedError):
            module.add_data('input', data=[
                ('text', "This is a text input"),
                ('dummy', "???")
            ])

    def test_clear_history(self):
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(model)

        module.add_data('input', [('text', "This is a test message 1.")])
        module.clear_history()
        self.assertEqual(module.history, [[]])
        self.assertEqual(module.cur_num_tokens_cache, [[]])

        module.add_data('input', [('text', "This is a test message 2.")])
        module.add_data('output', [('text', "This is a test response.")])
        module.add_data('input', data=[('image', np.random.randint(256, size=(100, 200, 4)).astype(np.uint8))])
        module.clear_history()
        self.assertEqual(module.history, [[]])
        self.assertEqual(module.cur_num_tokens_cache, [[]])

    def test_get_num_image_tokens(self):
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(model)

        self.assertEqual(module._get_num_image_tokens(128, 1024, 'low'), 85)
        self.assertEqual(module._get_num_image_tokens(4096, 4096, 'low'), 85)
        self.assertEqual(module._get_num_image_tokens(512, 64, 'low'), 85)
        self.assertEqual(module._get_num_image_tokens(128, 128), 255)
        self.assertEqual(module._get_num_image_tokens(512, 512), 255)
        self.assertEqual(module._get_num_image_tokens(600, 150), 425)
        self.assertEqual(module._get_num_image_tokens(130, 515), 425)
        self.assertEqual(module._get_num_image_tokens(900, 1000), 765)
        self.assertEqual(module._get_num_image_tokens(1280, 1080), 765)
        self.assertEqual(module._get_num_image_tokens(4096, 1920), 1445)
        self.assertEqual(module._get_num_image_tokens(1080, 4096), 1445)

    def test_get_message_num_tokens(self):
        system_prompt = "This is a test system prompt."
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(model)

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
        module._get_num_image_tokens = MagicMock(side_effect=side_effect_image)

        self.assertEqual(module._get_num_tokens('system', [{'type': 'text', 'text': system_prompt}]), 20)
        self.assertEqual(module._get_num_tokens('user', [{'type': 'text', 'text': "What's your name?"}]), 14)
        self.assertEqual(module._get_num_tokens('assistant', [{'type': 'text', 'text': "Live as if you were to die tomorrow."}]), 28)

        self.assertEqual(module._get_num_tokens(
            'user', content=[{'type': 'image_url', 'image_url': {'url': 'random-url'}} for i in range(3)], image_sizes=[(128, 128)] * 3), 76)
        self.assertEqual(module._get_num_tokens(
            'user', 
            content=[
                {'type': 'image_url', 'image_url': {'url': 'random-url'}},
                {'type': 'text', 'text': "Good job"}
            ],
            image_sizes=[(1024, 1024)]
        ), 137)
        self.assertEqual(module._get_num_tokens(
            'user',
            content=[
                {'type': 'image_url', 'image_url': {'url': "random-url-1"}},
                {'type': 'image_url', 'image_url': {'url': "random-url-2"}},
                {'type': 'image_url', 'image_url': {'url': "random-url-3"}},
                {'type': 'text', 'text': "What is your name?"},
                {'type': 'image_url', 'image_url': {'url': "random-url-4"}},
                {'type': 'text', 'text': "How are you doing today?"}
            ],
            image_sizes=[(640, 512), (128, 127), (1024, 1024), (64, 64)]
        ), 323)

    def test_find_starting_point(self):
        system_prompt = "This is a test system prompt."
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(model)
        module.cur_num_tokens_cache = [[] for _ in range(5)]
        for i in range(5):
            module.max_num_tokens = 1152
            module._get_num_tokens = MagicMock(return_value=10)
            module.cur_num_tokens_cache[i] = [100, 100, 100, 100, 100]
            self.assertEqual(module._find_starting_point(system_prompt, idx=i), 0)
            module.cur_num_tokens_cache[i] = [500, 200, 300, 400]
            self.assertEqual(module._find_starting_point(system_prompt, idx=i), 1)
            module.cur_num_tokens_cache[i] = [57, 100, 231, 10, 490, 126]
            self.assertEqual(module._find_starting_point(system_prompt, idx=i), 0)
            module.cur_num_tokens_cache[i] = [68, 367, 111, 77, 132, 2, 100, 501, 92]
            self.assertEqual(module._find_starting_point(system_prompt, idx=i), 3)

            module.max_num_tokens = 11
            module.cur_num_tokens_cache[i] = [10, 5, 18, 24]
            self.assertEqual(module._find_starting_point(system_prompt, idx=i), 4)

    def test_max_concurrent_prompts(self):
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(model, 3)
        self.assertEqual(module.max_concurrent_prompts, 3)
        module.history = [['test'], [2], [3]]
        module.cur_num_tokens_cache = [[10], [91], [18]]
        module.max_concurrent_prompts = 5
        self.assertEqual(module.max_concurrent_prompts, 5)
        self.assertEqual(len(module.history), 5)
        self.assertEqual(len(module.cur_num_tokens_cache), 5)
        module.max_concurrent_prompts = 2
        self.assertEqual(module.max_concurrent_prompts, 2)
        self.assertEqual(len(module.history), 2)
        self.assertEqual(len(module.cur_num_tokens_cache), 2)
        self.assertEqual(module.history[0], ['test'])
        self.assertEqual(module.history[1], [2])
        self.assertEqual(module.cur_num_tokens_cache[0], [10])
        self.assertEqual(module.cur_num_tokens_cache[1], [91])
        
        

if __name__=="__main__":
    unittest.main()
