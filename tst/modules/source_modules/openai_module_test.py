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
        self.assertEqual(module.history, [])
        self.assertEqual(module.cur_num_tokens_cache, [])
        self.assertEqual(module.encoding.name, 'o200k_base')

        model = "gpt-4"
        module = OpenAIModule(model)
        self.assertEqual(module.model, model)
        self.assertEqual(module.max_num_tokens, 8196)
        self.assertEqual(module.history, [])
        self.assertEqual(module.cur_num_tokens_cache, [])
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
            self.assertEqual(module.history[q*2], {'role': 'user', 'content': [{'type': 'text', 'text': queries[q]}]})
            self.assertEqual(module.history[q*2+1], {'role': 'assistant', 'content': [{'type': 'text', 'text': response}]})
            self.assertEqual(module.cur_num_tokens_cache[q*2], module._get_num_tokens('user', [{'type': 'text', 'text': query}]))
            self.assertEqual(module.cur_num_tokens_cache[q*2+1], module._get_num_tokens('assistant', [{'type': 'text', 'text': response}]))
            self.assertEqual(len(module.history), q*2+2)
            self.assertEqual(len(module.cur_num_tokens_cache), q*2+2)

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
            self.assertEqual(module.history[m*2], {'role': 'user', 'content': expected_content})
            self.assertEqual(module.history[m*2+1], {'role': 'assistant', 'content': [{'type': 'text', 'text': response}]})
            self.assertEqual(module.cur_num_tokens_cache[m*2], module._get_num_tokens('user', expected_content))
            self.assertEqual(module.cur_num_tokens_cache[m*2+1], module._get_num_tokens('assistant', [{'type': 'text', 'text': response}]))
            self.assertEqual(len(module.history), m*2+2)
            self.assertEqual(len(module.cur_num_tokens_cache), m*2+2)

    def test_batch_infer_with_texts(self):
        system_prompt = "This is a test system prompt."
        model = "gpt-4o-2024-05-13"
        
        # Single object queries
        module = OpenAIModule(model)
        queries = [
            "What's your name?",
            "What is the advantages of Python language?",
            "What is the definition of a generalist foundation model?"
        ]
        
        query_batches = [queries, queries[:1], queries[-2:]]
        prev_history_count = 0
        for qb, queries in enumerate(query_batches):
            mock_responses, model_inputs = [], []
            for q, query in enumerate(queries):
                mock_responses.append(f"This is a response {q} for batch {qb}.")
                
                model_inputs.append([('text', query)])
            
            module._get_batch_response_from_api = MagicMock(return_value=mock_responses[:])
            responses = module.batch_infer_step(model_inputs, system_prompt)
            
            self.assertEqual(responses, mock_responses)
            
            for q, query in enumerate(queries):
                query_content = [{'type': 'text', 'text': query}]
                self.assertEqual(module.history[prev_history_count+q], {'role': 'user', 'content': query_content})
                self.assertEqual(module.cur_num_tokens_cache[prev_history_count+q], module._get_num_tokens('user', query_content))
            
                resp_content = [{'type': 'text', 'text': responses[q]}]
                self.assertEqual(module.history[prev_history_count+q+len(queries)], {'role': 'assistant', 'content': resp_content})
                self.assertEqual(module.cur_num_tokens_cache[prev_history_count+q+len(queries)], module._get_num_tokens('assistant', resp_content))
            
            self.assertEqual(len(module.history), prev_history_count + len(queries) * 2)
            self.assertEqual(len(module.cur_num_tokens_cache), prev_history_count + len(queries) * 2)
            
            prev_history_count += len(queries) * 2
            
        # Multi-object queries
        system_prompt = "This is a test system prompt."
        model = "gpt-4o-2024-05-13"
        
        module = OpenAIModule(model)
        multi_queries = [
            ["What is the definition of a generalist foundation model?"],
            ["This is a user message 1", "Additional user message."],
            ["What's your name?", "My name is Python.", "Nice to meet you."]
        ]
        
        query_batches = [multi_queries, multi_queries[:1], multi_queries[-2:]]
        prev_history_count = 0
        for qb, queries in enumerate(query_batches):
            mock_responses, model_inputs = [], []
            for q, query in enumerate(queries):
                mock_response = f"This is a response {q} for batch {qb}."
                mock_responses.append(mock_response)
                
                model_inputs.append([('text', obj) for obj in query])
            
            module._get_batch_response_from_api = MagicMock(return_value=mock_responses[:])
            responses = module.batch_infer_step(model_inputs, system_prompt)
            
            self.assertEqual(responses, mock_responses)
            
            for q, query in enumerate(queries):
                query_content = [{'type': 'text', 'text': obj} for obj in query]
                self.assertEqual(module.history[prev_history_count+q], {'role': 'user', 'content': query_content})
                self.assertEqual(module.cur_num_tokens_cache[prev_history_count+q], module._get_num_tokens('user', query_content))
            
                resp_content = [{'type': 'text', 'text': responses[q]}]
                self.assertEqual(module.history[prev_history_count+q+len(queries)], {'role': 'assistant', 'content': resp_content})
                self.assertEqual(module.cur_num_tokens_cache[prev_history_count+q+len(queries)], module._get_num_tokens('assistant', resp_content))
            
            self.assertEqual(len(module.history), prev_history_count + len(queries) * 2)
            self.assertEqual(len(module.cur_num_tokens_cache), prev_history_count + len(queries) * 2)
            
            prev_history_count += len(queries) * 2
    
    def test_batch_infer_with_multiple_system_prompts(self):
        model = "gpt-4o-2024-05-13"
        module = OpenAIModule(model)
        
        # Single object queries with multiple system prompts
        queries = [
            "What's your name?",
            "What is the advantages of Python language?",
            "What is the definition of a generalist foundation model?"
        ]
        
        system_prompts = [
            'This is a test system prompt 1.', 
            'This is a test system prompt 2.', 
            'This is a test system prompt 3.'
        ]
        
        query_batches = [queries, queries[:1], queries[-2:]]
        system_prompts_batches = [system_prompts, system_prompts[:1], system_prompts[-2:]]
        
        prev_history_count = 0
        for qb, queries in enumerate(query_batches):
            mock_responses, model_inputs = [], []
            for q, query in enumerate(queries):
                mock_responses.append(f"This is a response {q} for batch {qb}.")
                
                model_inputs.append([('text', query)])
            
            module._get_batch_response_from_api = MagicMock(return_value=mock_responses[:])
            responses = module.batch_infer_step(model_inputs, system_prompts_batches[qb])
            
            self.assertEqual(responses, mock_responses)
            
            for q, query in enumerate(queries):
                query_content = [{'type': 'text', 'text': query}]
                self.assertEqual(module.history[prev_history_count+q], {'role': 'user', 'content': query_content})
                self.assertEqual(module.cur_num_tokens_cache[prev_history_count+q], module._get_num_tokens('user', query_content))
            
                resp_content = [{'type': 'text', 'text': responses[q]}]
                self.assertEqual(module.history[prev_history_count+q+len(queries)], {'role': 'assistant', 'content': resp_content})
                self.assertEqual(module.cur_num_tokens_cache[prev_history_count+q+len(queries)], module._get_num_tokens('assistant', resp_content))
            
            self.assertEqual(len(module.history), prev_history_count + len(queries) * 2)
            self.assertEqual(len(module.cur_num_tokens_cache), prev_history_count + len(queries) * 2)
            
            prev_history_count += len(queries) * 2
            
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
            self.assertEqual(module.history[q*2+1], {'role': 'assistant', 'content': [{'type': 'text', 'text': response}]})
            self.assertEqual(module.history[q*2]['role'], 'user')
            self.assertTrue(isinstance(module.history[q*2]['content'], list))

            for o, obj in enumerate(query):
                if obj[0] == 'text':
                    self.assertEqual(module.history[q*2]['content'][o], {'type': 'text', 'text': obj[1]})
                elif obj[0] == 'image':
                    self.assertEqual(module.history[q*2]['content'][o]['type'], 'image_url')
                    self.assertTrue(isinstance(module.history[q*2]['content'][o]['image_url']['url'], str))
                    self.assertTrue(module.history[q*2]['content'][o]['image_url']['url'].startswith('data:image/png;base64,'))

            self.assertEqual(module.cur_num_tokens_cache[q*2], module._get_num_tokens(
                'user',
                 module.history[q*2]['content'], 
                [(obj[1].shape[0], obj[1].shape[1]) for obj in query if obj[0] == 'image']
            ))
            self.assertEqual(module.cur_num_tokens_cache[q*2+1], module._get_num_tokens('assistant', [{'type': 'text', 'text': response}]))

            self.assertEqual(len(module.history), q*2+2)
            self.assertEqual(len(module.cur_num_tokens_cache), q*2+2)

    def test_batch_infer_with_images(self):
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

        query_batches = [queries, queries[:2], queries[:1]]
        prev_history_count = 0
        for qb, queries in enumerate(query_batches):
            mock_responses, model_inputs = [], []
            for q, query in enumerate(queries):
                mock_response = f"This is a response {q} for batch {qb}."
                mock_responses.append(mock_response)
                
                model_inputs.append(query)
            module._get_batch_response_from_api = MagicMock(return_value=mock_responses[:])
            responses = module.batch_infer_step(model_inputs, system_prompt)
            
            self.assertEqual(responses, mock_responses)

            for q, query in enumerate(queries):
                self.assertEqual(module.history[prev_history_count+q]['role'], 'user')
                self.assertTrue(isinstance(module.history[prev_history_count+q]['content'], list))
                
                shapes = []
                for o, obj in enumerate(query):
                    if obj[0] == 'text':
                        self.assertEqual(module.history[prev_history_count+q]['content'][o], 
                                         {'type': 'text', 'text': obj[1]})
                    elif obj[0] == 'image':
                        obj_history_content = module.history[prev_history_count+q]['content'][o]
                        self.assertEqual(obj_history_content['type'], 'image_url')

                        img_url = obj_history_content['image_url']['url']
                        self.assertTrue(isinstance(img_url, str))
                        self.assertTrue(img_url.startswith('data:image/png;base64,'))
                    
                        shapes.append((obj[1].shape[0], obj[1].shape[1]))

                self.assertEqual(module.cur_num_tokens_cache[prev_history_count+q], 
                                 module._get_num_tokens('user', 
                                                        module.history[prev_history_count+q]['content'], 
                                                        shapes))                                

                resp_content = [{'type': 'text', 'text': responses[q]}]
                self.assertEqual(module.history[prev_history_count+q+len(queries)], {'role': 'assistant', 'content': resp_content})
                self.assertEqual(module.cur_num_tokens_cache[prev_history_count+q+len(queries)], module._get_num_tokens('assistant', resp_content))
            
            self.assertEqual(len(module.history), prev_history_count + len(queries) * 2)
            self.assertEqual(len(module.cur_num_tokens_cache), prev_history_count + len(queries) * 2)
            
            prev_history_count += len(queries) * 2

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
        self.assertEqual(module.history, [])
        self.assertEqual(module.cur_num_tokens_cache, [])

        module.add_data('input', [('text', "This is a test message 2.")])
        module.add_data('output', [('text', "This is a test response.")])
        module.add_data('input', data=[('image', np.random.randint(256, size=(100, 200, 4)).astype(np.uint8))])
        module.clear_history()
        self.assertEqual(module.history, [])
        self.assertEqual(module.cur_num_tokens_cache, [])

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
        module.max_num_tokens = 1152

        module._get_num_tokens = MagicMock(return_value=10)
        module.cur_num_tokens_cache = [100, 100, 100, 100, 100]
        self.assertEqual(module._find_starting_point(system_prompt), 0)
        module.cur_num_tokens_cache = [500, 200, 300, 400]
        self.assertEqual(module._find_starting_point(system_prompt), 1)
        module.cur_num_tokens_cache = [57, 100, 231, 10, 490, 126]
        self.assertEqual(module._find_starting_point(system_prompt), 0)
        module.cur_num_tokens_cache = [68, 367, 111, 77, 132, 2, 100, 501, 92]
        self.assertEqual(module._find_starting_point(system_prompt), 3)

        module.max_num_tokens = 11
        module.cur_num_tokens_cache = [10, 5, 18, 24]
        self.assertEqual(module._find_starting_point(system_prompt), 4)


if __name__=="__main__":
    unittest.main()
