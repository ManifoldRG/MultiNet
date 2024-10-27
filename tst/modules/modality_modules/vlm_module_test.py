import os
import sys
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
os.environ["OPENAI_API_KEY"] = "random-api-key"

from src.modules.modality_modules.vlm_module import VLMModule
from src.modules.source_modules.openai_module import OpenAIModule
from unittest.mock import MagicMock, call

import unittest
import numpy as np


class VLMModuleTest(unittest.TestCase):
    def test_constructor(self):
        # Testing OpenAIModule.
        source = 'openai'
        model = 'gpt-4o-2024-05-13'
        module = VLMModule(source, model)
        self.assertTrue(isinstance(module.source_module, OpenAIModule))

        # TODO: Add any source module constructor test after newly implemented.
        
        # Testing constructor failure.
        model = 'dummy-model'
        with self.assertRaises(KeyError):
            module = VLMModule(source, model)

    def test_convert_into_text(self):
        source = 'openai'
        model = 'gpt-4o-2024-05-13'
        module = VLMModule(source, model)

        self.assertEqual(module._convert_into_text('key1', "This is a value."), "key1: This is a value.")
        self.assertEqual(module._convert_into_text('key2', [1, 2, 3, 'v1', 'v2']), "key2: [1, 2, 3, 'v1', 'v2']")
        self.assertEqual(module._convert_into_text('key3', {'k1': 'v1', 'k2': 16, 'k3': [10, 20, '30']}), "key3: {'k1': 'v1', 'k2': 16, 'k3': [10, 20, '30']}")
        self.assertEqual(module._convert_into_text('key4', np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])), "key4: [[ 1  2  3  4]\n [ 5  6  7  8]\n [ 9 10 11 12]]")
        self.assertEqual(module._convert_into_text('key5', 100.2478), "key5: 100.2478")

    def text_convert_into_data(self):
        source = 'openai'
        model = 'gpt-4o-2024-05-13'
        module = VLMModule(source, model)

        self.assertEqual(module._convert_into_data("1", int), 1)
        self.assertEqual(module._convert_into_data("46", int), 46)
        self.assertEqual(module._convert_into_data("2.51245", float), 2.51245)
        self.assertEqual(module._convert_into_data("100.00", float), 100.00)
        self.assertEqual(module._convert_into_data("[1,2,3,4,5,6]", list), [1, 2, 3, 4, 5, 6])
        self.assertEqual(module._convert_into_data("[1, 2.4, 6.73, 10]", list), [1, 2.4, 6.73, 10])
        self.assertEqual(module._convert_into_data("0 11.04957"), (0, 11.04957))
        self.assertEqual(module._convert_into_data("8 100.09809"), (8, 100.09809))
        with self.assertRaises(ValueError):
            module._convert_into_data("46uin", int)
        with self.assertRaises(SyntaxError):
            module._convert_into_data("[1,5,2,09\fgjg]fs", list)
        with self.assertRaises(ValueError):
            module._convert_into_data("grgj84u982fu", tuple)

    def test_process_inputs_zero_shot(self):
        source = 'openai'
        model = 'gpt-4o-2024-05-13'
        module = VLMModule(source, model)

        cur_inputs = [
            [
                ('image_observation', np.random.randint(256, size=(100, 40, 3)).astype(np.uint8)),
                ('image_observation', np.random.randint(256, size=(100, 40, 3)).astype(np.uint8)),
                ('text_observation', "This is text instruction 1.")
            ],
            [
                ('image_observation', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('image_observation', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('image_observation', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('text_observation', "This is text instruction 2."),
                ('discrete_observation', np.random.randint(2, size=(12)))
            ],
            [
                ('image_observation', np.random.randint(256, size=(64, 64, 3)).astype(np.uint8))
            ],
            [
                ('reward', "100.45865"),
                ('image_observation', np.random.randint(256, size=(1024, 1024, 3)).astype(np.uint8)),
                ('image_observation', np.random.randint(256, size=(1024, 1024, 3)).astype(np.uint8)),
                ('continuous_observation', np.random.random_sample(size=(8)))
            ]
        ]
        processed_cur_inputs, processed_k_shots_examples = module._process_inputs(cur_inputs)

        self.assertEqual(processed_k_shots_examples, [])
        self.assertEqual(len(processed_cur_inputs), len(cur_inputs))

        for b in range(len(cur_inputs)):
            self.assertEqual(len(processed_cur_inputs[b]), len(cur_inputs[b]))
            for i in range(len(cur_inputs[b])):
                if cur_inputs[b][i][0].startswith('image_'):
                    self.assertEqual(processed_cur_inputs[b][i], ('image', cur_inputs[b][i][1]))
                else:
                    self.assertEqual(processed_cur_inputs[b][i], ('text', f"{cur_inputs[b][i][0]}: {cur_inputs[b][i][1]}"))

    def test_process_inputs_one_shot(self):
        source = 'openai'
        model = 'gpt-4o-2024-05-13'
        module = VLMModule(source, model)

        cur_inputs = [
            [
                ('image_observation', np.random.randint(256, size=(256, 256, 4)).astype(np.uint8)),
                ('image_observation', np.random.randint(256, size=(100, 40, 4)).astype(np.uint8)),
                ('text_observation', "Hi, how are you?"),
                ('text_observation', "This is an instruction.")
            ],
            [
                ('image_observation', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('image_observation', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('image_observation', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('continuous_observation', np.random.random_sample(size=(32))),
                ('discrete_observation', np.random.randint(2, size=(12)))
            ],
            [
                ('discrete_observation', np.random.randint(4, size=(8))),
                ('image_observation', np.random.randint(256, size=(120, 80, 4)).astype(np.uint8))
            ]
        ]
        k_shots_examples = [
            [
                [
                    ('input', [
                        ('image_observation', np.random.randint(256, size=(64, 64, 4)).astype(np.uint8)),
                        ('text_observation', "This is the first demonstration."),
                        ('text_observation', "Check carefully how it is solved.")
                    ]),
                    ('output', "This is example output 1.")
                ]
            ],
            [
                [
                    ('input', [
                        ('image_observation', np.random.randint(256, size=(512, 512, 3)).astype(np.uint8)),
                        ('image_observation', np.random.randint(256, size=(512, 512, 3)).astype(np.uint8)),
                        ('image_observation', np.random.randint(256, size=(512, 512, 3)).astype(np.uint8)),
                        ('continuous_observation', np.random.random_sample(size=(32))),
                        ('discrete_observation', np.random.randint(2, size=(12)))
                    ]),
                    ('output', "This is example output 2.")
                ]
            ],
            [
                [
                    ('input', [
                        ('discrete_observation', np.random.randint(4, size=(8))),
                        ('image_observation', np.random.randint(256, size=(120, 80, 4)))
                    ]),
                    ('output', str(np.random.randint(8, size=(6)))),
                    ('input', [
                        ('reward', "24.102")
                    ])
                ]
            ]
        ]
        processed_cur_inputs, processed_k_shots_examples = module._process_inputs(cur_inputs, k_shots_examples)

        self.assertEqual(len(processed_cur_inputs), len(cur_inputs))
        self.assertEqual(len(processed_k_shots_examples), len(k_shots_examples))

        for b in range(len(cur_inputs)):
            self.assertEqual(len(processed_cur_inputs[b]), len(cur_inputs[b]))
            for i in range(len(cur_inputs[b])):
                if cur_inputs[b][i][0].startswith('image_'):
                    self.assertEqual(processed_cur_inputs[b][i], ('image', cur_inputs[b][i][1]))
                else:
                    self.assertEqual(processed_cur_inputs[b][i], ('text', f"{cur_inputs[b][i][0]}: {cur_inputs[b][i][1]}"))

        for b in range(len(k_shots_examples)):
            self.assertEqual(len(processed_k_shots_examples[b]), len(k_shots_examples[b]))
            num_examples = len(k_shots_examples[b])
            for k in range(num_examples):
                self.assertEqual(len(processed_k_shots_examples[b][k]), len(k_shots_examples[b][k]))
                for i in range(len(k_shots_examples[b][k])):
                    if k_shots_examples[b][k][i][0] == 'output':
                        self.assertEqual(processed_k_shots_examples[b][k][i], k_shots_examples[b][k][i][1])
                    else:
                        expected_data = []
                        for type, value in k_shots_examples[b][k][i][1]:
                            if type.startswith('image_'):
                                expected_data.append(('image', value))
                            else:
                                expected_data.append(('text', f"{type}: {value}"))
                        self.assertEqual(processed_k_shots_examples[b][k][i], expected_data)

    def test_process_inputs_three_shots(self):
        source = 'openai'
        model = 'gpt-4o-2024-05-13'
        module = VLMModule(source, model)

        cur_inputs = [
            [
                ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                ('text_observation', "Describe the pictures."),
            ],
            [
                ('image_observation', np.random.randint(256, size=(1024, 1024, 3)).astype(np.uint8)),
                ('image_observation', np.random.randint(256, size=(512, 764, 2)).astype(np.uint8)),
                ('continuous_observation', np.random.random_sample(size=(20))),
                ('discrete_observation', np.random.randint(4, size=(16)))
            ],
            [
                ('discrete_observation', np.random.randint(2, size=(100))),
                ('image_observation', np.random.randint(256, size=(24, 24, 4)).astype(np.uint8)),
                ('image_observation', np.random.randint(256, size=(24, 24, 4)).astype(np.uint8))
            ],
            [
                ('image_observation', np.random.randint(256, size=(48, 96, 3)).astype(np.uint8)),
                ('text_observation', "Solve this puzzle."),
                ('continuous_observation', np.random.random_sample(size=(10)))
            ]
        ]
        k_shots_examples = [
            [
                [
                    ('input', [
                        ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                        ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                        ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                        ('text_observation', "Describe the pictures (example 1)."),
                    ]),
                    ('output', "This is example output 1.")
                ],
                [
                    ('input', [
                        ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                        ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                        ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                        ('text_observation', "Describe the pictures (example 2)."),
                    ]),
                    ('output', "This is example output 2.")
                ],
                [
                    ('input', [
                        ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                        ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                        ('image_observation', np.random.randint(256, size=(160, 160, 4)).astype(np.uint8)),
                        ('text_observation', "Describe the pictures (example 3)."),
                    ]),
                    ('output', "This is example output 3.")
                ],
            ],
            [
                [
                    ('input', [
                        ('image_observation', np.random.randint(256, size=(1024, 1024, 3)).astype(np.uint8)),
                        ('image_observation', np.random.randint(256, size=(512, 764, 4)).astype(np.uint8)),
                        ('continuous_observation', np.random.random_sample(size=(20))),
                        ('discrete_observation', np.random.randint(4, size=(16)))
                    ]),
                    ('output', "The correct action is 1."),
                    ('input', [
                        ('reward', "120.64")
                    ])
                ],
                [
                    ('input', [
                        ('image_observation', np.random.randint(256, size=(1024, 1024, 3)).astype(np.uint8)),
                        ('image_observation', np.random.randint(256, size=(512, 764, 4)).astype(np.uint8)),
                        ('continuous_observation', np.random.random_sample(size=(20))),
                        ('discrete_observation', np.random.randint(4, size=(16)))
                    ]),
                    ('output', "The correct action is 1."),
                    ('input', [
                        ('reward', "-10.99")
                    ])
                ],
            ],
            [
                [
                    ('input', [
                        ('discrete_observation', np.random.randint(2, size=(100))),
                        ('image_observation', np.random.randint(256, size=(24, 24, 4)).astype(np.uint8)),
                        ('image_observation', np.random.randint(256, size=(24, 24, 4)).astype(np.uint8))
                    ]),
                    ('output', str(np.random.randint(4, size=(3)))),
                    ('input', [
                        ('reward', "8.3156467")
                    ])
                ],
            ],
            [
                []
            ]
        ]
        processed_cur_inputs, processed_k_shots_examples = module._process_inputs(cur_inputs, k_shots_examples)

        self.assertEqual(len(processed_cur_inputs), len(cur_inputs))
        self.assertEqual(len(processed_k_shots_examples), len(k_shots_examples))

        for b in range(len(cur_inputs)):
            self.assertEqual(len(processed_cur_inputs[b]), len(cur_inputs[b]))
            for i in range(len(cur_inputs[b])):
                if cur_inputs[b][i][0].startswith('image_'):
                    self.assertEqual(processed_cur_inputs[b][i], ('image', cur_inputs[b][i][1]))
                else:
                    self.assertEqual(processed_cur_inputs[b][i], ('text', f"{cur_inputs[b][i][0]}: {cur_inputs[b][i][1]}"))

        for b in range(len(k_shots_examples)):
            self.assertEqual(len(processed_k_shots_examples[b]), len(k_shots_examples[b]))
            num_examples = len(k_shots_examples[b])
            for k in range(num_examples):
                self.assertEqual(len(processed_k_shots_examples[b][k]), len(k_shots_examples[b][k]))
                for i in range(len(k_shots_examples[b][k])):
                    if k_shots_examples[b][k][i][0] == 'output':
                        self.assertEqual(processed_k_shots_examples[b][k][i], k_shots_examples[b][k][i][1])
                    else:
                        expected_data = []
                        for type, value in k_shots_examples[b][k][i][1]:
                            if type.startswith('image_'):
                                expected_data.append(('image', value))
                            else:
                                expected_data.append(('text', f"{type}: {value}"))
                        self.assertEqual(processed_k_shots_examples[b][k][i], expected_data)

    def test_infer_step_with_openai(self):
        source = 'openai'
        model = 'gpt-4o-2024-05-13'
        module = VLMModule(source, model)
        system_prompt = "This is a test prompt."

        # Zero-shot test.
        processed_cur_inputs = [
            [
                ('image', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(128, 128, 3)).astype(np.uint8)),
                ('text', "text_observation: This is a text description."),
                ('text', f"continuous_observation: {np.random.random_sample(size=(12))}")
            ],
            [
                ('image', np.random.randint(256, size=(50, 100, 3)).astype(np.uint8)),
                ('text', f"discrete_observation: {np.random.randint(8, size=(10))}")
            ]
        ]
        module._process_inputs = MagicMock(return_value=(processed_cur_inputs, []))
        def side_effect(input_data, system_prompt):
            if input_data == processed_cur_inputs[0]:
                return "Correctly got the first data for zero-shot."
            elif input_data == processed_cur_inputs[1]:
                return "Correctly got the second data for zero-shot."
            return None
        module.source_module.infer_step = MagicMock(side_effect=side_effect)
        outputs = module.infer_step([], [], [system_prompt for i in range(2)])
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0], "Correctly got the first data for zero-shot.")
        self.assertEqual(outputs[1], "Correctly got the second data for zero-shot.")

        # Few-shot test.
        processed_cur_inputs = [
            [
                ('image', np.random.randint(256, size=(256, 256, 3)).astype(np.uint8)),
                ('image', np.random.randint(256, size=(20, 20, 4)).astype(np.uint8)),
                ('text', f"discrete_observation: {np.random.randint(12, size=(24))}")
            ],
            [
                ('image', np.random.randint(256, size=(240, 240, 4)).astype(np.uint8)),
                ('text', f"text_observation: This is additional text description."),
                ('image', np.random.randint(256, size=(64, 128, 3)).astype(np.uint8)),
                ('text', f"continuous_observation: {np.random.random_sample(size=(100))}")
            ],
            [
                ('image', np.random.randint(256, size=(100, 100, 4))),
            ]
        ]
        processed_k_shots_examples = [
            [
                [
                    [
                        ('image', np.random.randint(256, size=(256, 256, 3)).astype(np.uint8)),
                        ('image', np.random.randint(256, size=(20, 20, 4)).astype(np.uint8)),
                        ('text', f"discrete_observation: {np.random.randint(12, size=(24))}")
                    ],
                    "The sample output 1 for batch 0."
                ],
                [
                    [
                        ('image', np.random.randint(256, size=(256, 256, 3)).astype(np.uint8)),
                        ('image', np.random.randint(256, size=(20, 20, 4)).astype(np.uint8)),
                        ('text', f"discrete_observation: {np.random.randint(12, size=(24))}")
                    ],
                    "The sample output 2 for batch 0."
                ]
            ],
            [
                [
                    [
                        ('image', np.random.randint(256, size=(240, 240, 4)).astype(np.uint8)),
                        ('text', f"text_observation: This is additional text description."),
                        ('image', np.random.randint(256, size=(64, 128, 3)).astype(np.uint8)),
                        ('text', f"continuous_observation: {np.random.random_sample(size=(100))}")
                    ],
                    "The sample output 1 for batch 1.",
                    [
                        ('text', "reward: 100.00")
                    ]
                ],
                [
                    [
                        ('image', np.random.randint(256, size=(240, 240, 4)).astype(np.uint8)),
                        ('text', f"text_observation: This is additional text description."),
                        ('image', np.random.randint(256, size=(64, 128, 3)).astype(np.uint8)),
                        ('text', f"continuous_observation: {np.random.random_sample(size=(100))}")
                    ],
                    "The sample output 2 for batch 1.",
                    [
                        ('text', "reward: 18.902")
                    ]
                ]
            ],
            [
                [
                    [
                        ('image', np.random.randint(256, size=(100, 100, 4)).astype(np.uint8)),
                    ],
                    "The sample output 3 for batch 2."
                ]
            ]
        ]
        module._process_inputs = MagicMock(return_value=(processed_cur_inputs, processed_k_shots_examples))
        def side_effect(input_data, system_prompt):
            if input_data == processed_cur_inputs[0]:
                return "Correctly got the first data for 2-shots."
            elif input_data == processed_cur_inputs[1]:
                return "Correctly got the second data for 2-shots."
            elif input_data == processed_cur_inputs[2]:
                return "Correctly got the third data for 2-shots."
            return None
        module.source_module.infer_step = MagicMock(side_effect=side_effect)
        module.source_module.add_data = MagicMock()

        outputs = module.infer_step([], [], [system_prompt for i in range(3)])
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0], "Correctly got the first data for 2-shots.")
        self.assertEqual(outputs[1], "Correctly got the second data for 2-shots.")
        self.assertEqual(outputs[2], "Correctly got the third data for 2-shots.")
        module.source_module.add_data.assert_has_calls([
            call('input', processed_k_shots_examples[0][0][0]),
            call('output', [('text', processed_k_shots_examples[0][0][1])]),
            call('input', processed_k_shots_examples[0][1][0]),
            call('output', [('text', processed_k_shots_examples[0][1][1])]),
            call('input', processed_k_shots_examples[1][0][0]),
            call('output', [('text', processed_k_shots_examples[1][0][1])]),
            call('input', processed_k_shots_examples[1][0][2]),
            call('input', processed_k_shots_examples[1][1][0]),
            call('output', [('text', processed_k_shots_examples[1][1][1])]),
            call('input', processed_k_shots_examples[1][1][2]),
            call('input', processed_k_shots_examples[2][0][0]),
            call('output', [('text', processed_k_shots_examples[2][0][1])])
        ])


if __name__=="__main__":
    unittest.main()
