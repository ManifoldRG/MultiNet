import os
import sys
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

from src.modules.dataset_modules.openx_module import OpenXModule
from definitions.prompt import format_instruction_prompt
from definitions.openx import DESCRIPTIONS, ACTION_SPACES, ACTION_EXCLUSIVENESS, ADDITIONAL_INSTRUCTIONS
from unittest.mock import MagicMock

import unittest
import random
import numpy as np


class OpenXModuleTest(unittest.TestCase):
    def test_constructor(self):
        # Test for VLMModule with OpenAIModule.
        disk_root_dir = "/mnt/disks"
        modality = "vlm"
        source = "openai"
        model = 'gpt-4o-2024-05-13'
        batch_size = 1
        k_shots = 2
        module = OpenXModule(disk_root_dir, modality, source, model, batch_size, k_shots)
        self.assertEqual(module.disk_root_dir, disk_root_dir)
        self.assertEqual(module.modality, modality)
        self.assertEqual(module.source, source)
        self.assertEqual(module.model, model)
        self.assertEqual(module.batch_size, batch_size)
        self.assertEqual(module.k_shots, k_shots)

        # TODO: Add any source module constructor test after newly implemented.

    def test_process_batch_zero_shot_1(self):
        batch = {
            'continuous_observation': [
                [np.random.random(size=(16)) for i in range(4)],
                [np.random.random(size=(16)) for i in range(1)],
                [np.random.random(size=(16)) for i in range(3)],
                [np.random.random(size=(16)) for i in range(5)],
            ],
            'text_observation': [
                ['test-text1' for i in range(4)],
                ['test-text1' for i in range(1)],
                ['test-text2' for i in range(3)],
                ['test-text2' for i in range(5)],
            ],
            'image_observation': [
                [np.random.randint(256, size=(128,128,3)) for i in range(4)],
                [np.random.randint(256, size=(128,128,3)) for i in range(1)],
                [np.random.randint(256, size=(128,128,3)) for i in range(3)],
                [np.random.randint(256, size=(128,128,3)) for i in range(5)],
            ],
            'discrete_observation': None,
            'action': [
                [np.random.random(size=(8)) for i in range(4)],
                [np.random.random(size=(8)) for i in range(1)],
                [np.random.random(size=(8)) for i in range(3)],
                [np.random.random(size=(8)) for i in range(5)],
            ],
            'reward': [
                [random.random() for i in range(4)],
                [random.random() for i in range(1)],
                [random.random() for i in range(3)],
                [random.random() for i in range(5)]
            ],
            'is_last': [
                [False, False, False, True],
                [True],
                [False, False, True],
                [False, False, False, False, True]
            ]
        }

        module = OpenXModule(
            disk_root_dir='/mnt/disks', 
            modality='vlm', 
            source='openai',
            model='gpt-4o-2024-05-13', 
            batch_size=4, 
            k_shots=0
        )
        dataset = 'test-dataset'

        def side_effect(dataset, env_name):
            if env_name == 'test-text1':
                return "test-instruction-1"
            elif env_name == 'test-text2':
                return "test-instruction-2"
            else:
                return "dummy-instruction"
        module._get_vlm_instruction = MagicMock(side_effect=side_effect)
        module._get_output_type = MagicMock(return_value=list)

        for i, (cur_inputs, k_shots_examples, instructions, labels, idxs, output_types) in enumerate(module._process_batch(batch, dataset)):
            self.assertEqual(k_shots_examples, [])
            if 0 <= i < 1:  # Batch 0, 1, 2, 3
                self.assertEqual(idxs, [0, 1, 2, 3])
                self.assertEqual(instructions, ["test-instruction-1", "test-instruction-1", "test-instruction-2", "test-instruction-2"])
                self.assertEqual(labels, [batch['action'][b][i] for b in [0, 1, 2, 3]])
                self.assertEqual(output_types, [list, list, list, list])
                self.assertEqual(cur_inputs, [
                    [('image_observation', batch['image_observation'][b][i]), ('continuous_observation', batch['continuous_observation'][b][i])] for b in [0, 1, 2, 3]
                ])

            elif 1 <= i < 3:  # Batch 0, 2, 3
                self.assertEqual(idxs, [0, 2, 3])
                self.assertEqual(instructions, ["test-instruction-1", "test-instruction-2", "test-instruction-2"])
                self.assertEqual(labels, [batch['action'][b][i] for b in [0, 2, 3]])
                self.assertEqual(output_types, [list, list, list])
                self.assertEqual(cur_inputs, [
                    [('image_observation', batch['image_observation'][b][i]), ('continuous_observation', batch['continuous_observation'][b][i])] for b in [0, 2, 3]
                ])
            elif 3 <= i < 4:  # Batch 0, 3
                self.assertEqual(idxs, [0, 3])
                self.assertEqual(instructions, ["test-instruction-1", "test-instruction-2"])
                self.assertEqual(labels, [batch['action'][b][i] for b in [0, 3]])
                self.assertEqual(output_types, [list, list])
                self.assertEqual(cur_inputs, [
                    [('image_observation', batch['image_observation'][b][i]), ('continuous_observation', batch['continuous_observation'][b][i])] for b in [0, 3]
                ])
            else:  # Batch 3
                self.assertEqual(idxs, [3])
                self.assertEqual(instructions, ["test-instruction-2"])
                self.assertEqual(labels, [batch['action'][b][i] for b in [3]])
                self.assertEqual(output_types, [list])
                self.assertEqual(cur_inputs, [
                    [('image_observation', batch['image_observation'][b][i]), ('continuous_observation', batch['continuous_observation'][b][i])] for b in [3]
                ])
        self.assertEqual(i, 4)  # Stopping check.

    def test_process_batch_zero_shot_2(self):
        batch = {
            'continuous_observation': [
                [np.random.random(size=(32)) for i in range(10)],
                [np.random.random(size=(32)) for i in range(21)],
                [np.random.random(size=(32)) for i in range(8)],
                [np.random.random(size=(32)) for i in range(32)],
                [np.random.random(size=(32)) for i in range(8)],
                [np.random.random(size=(32)) for i in range(12)],
                [np.random.random(size=(32)) for i in range(12)],
                [np.random.random(size=(32)) for i in range(24)]
            ],
            'text_observation': [
                ['test-text1' for i in range(10)],
                ['test-text1' for i in range(21)],
                ['test-text2' for i in range(8)],
                ['test-text2' for i in range(32)],
                ['test-text2' for i in range(8)],
                ['test-text3' for i in range(12)],
                ['test-text3' for i in range(12)],
                ['test-text3' for i in range(24)],
            ],
            'image_observation': [
                [np.random.randint(256, size=(640,320,3)) for i in range(10)],
                [np.random.randint(256, size=(4,640,320,3)) for i in range(21)],
                [np.random.randint(256, size=(640,320,3)) for i in range(8)],
                [np.random.randint(256, size=(640,320,3)) for i in range(32)],
                [np.random.randint(256, size=(640,320,3)) for i in range(8)],
                [np.random.randint(256, size=(2,640,320,3)) for i in range(12)],
                [np.random.randint(256, size=(640,320,3)) for i in range(12)],
                [np.random.randint(256, size=(640,320,3)) for i in range(24)],
            ],
            'discrete_observation': None,
            'action': [
                [np.random.random(size=(16)) for i in range(10)],
                [np.random.random(size=(16)) for i in range(21)],
                [np.random.random(size=(16)) for i in range(8)],
                [np.random.random(size=(16)) for i in range(32)],
                [np.random.random(size=(16)) for i in range(8)],
                [np.random.random(size=(16)) for i in range(12)],
                [np.random.random(size=(16)) for i in range(12)],
                [np.random.random(size=(16)) for i in range(24)],
            ],
            'reward': [
                [random.random() for i in range(10)],
                [random.random() for i in range(21)],
                [random.random() for i in range(8)],
                [random.random() for i in range(32)],
                [random.random() for i in range(8)],
                [random.random() for i in range(12)],
                [random.random() for i in range(12)],
                [random.random() for i in range(24)],
            ],
            'is_last': [
                [False for i in range(9)] + [True],
                [False for i in range(20)] + [True],
                [False for i in range(7)] + [True],
                [False for i in range(31)] + [True],
                [False for i in range(7)] + [True],
                [False for i in range(11)] + [True],
                [False for i in range(11)] + [True],
                [False for i in range(23)] + [True]
            ]
        }

        module = OpenXModule(
            disk_root_dir='/mnt/disks', 
            modality='vlm', 
            source='openai',
            model='gpt-4o-2024-05-13', 
            batch_size=8,
            k_shots=0
        )
        dataset = 'test-dataset'

        def side_effect(dataset, env_name):
            if env_name == 'test-text1':
                return "test-instruction-1"
            elif env_name == 'test-text2':
                return "test-instruction-2"
            else:
                return "dummy-instruction"
        module._get_vlm_instruction = MagicMock(side_effect=side_effect)
        module._get_output_type = MagicMock(return_value=list)

        for i, (cur_inputs, k_shots_examples, instructions, labels, idxs, output_types) in enumerate(module._process_batch(batch, dataset)):
            self.assertEqual(k_shots_examples, [])
            if 0 <= i < 8:  # Batch 0, 1, 2, 3, 4, 5, 6, 7
                self.assertEqual(idxs, [0, 1, 2, 3, 4, 5, 6, 7])
                self.assertEqual(instructions, [
                    "test-instruction-1", 
                    "test-instruction-1", 
                    "test-instruction-2", 
                    "test-instruction-2", 
                    "test-instruction-2", 
                    "dummy-instruction", 
                    "dummy-instruction", 
                    "dummy-instruction"
                ])
                self.assertEqual(labels, [batch['action'][b][i] for b in [0, 1, 2, 3, 4, 5, 6, 7]])
                self.assertEqual(output_types, [list, list, list, list, list, list, list, list])
                expected_cur_inputs = []
                for b in [0, 1, 2, 3, 4, 5, 6, 7]:
                    expected_cur_inputs.append([])
                    if len(batch['image_observation'][b][i].shape) == 3:
                        expected_cur_inputs[-1].append(('image_observation', batch['image_observation'][b][i]))
                    else:
                        expected_cur_inputs[-1] += [('image_observation', image) for image in batch['image_observation'][b][i]]
                    expected_cur_inputs[-1].append(('continuous_observation', batch['continuous_observation'][b][i]))

                self.assertEqual(len(cur_inputs), len(expected_cur_inputs))
                for b in range(len(expected_cur_inputs)):
                    batch_cur_inputs = cur_inputs[b]
                    batch_expected_cur_inputs = expected_cur_inputs[b]
                    self.assertEqual(len(batch_cur_inputs), len(batch_expected_cur_inputs))
                    for j in range(len(batch_expected_cur_inputs)):
                        self.assertEqual(batch_cur_inputs[j][0], batch_expected_cur_inputs[j][0])
                        self.assertTrue(np.array_equal(batch_cur_inputs[j][1], batch_expected_cur_inputs[j][1]))
            elif 8 <= i < 10:  # Batch 0, 1, 3, 5, 6, 7
                self.assertEqual(idxs, [0, 1, 3, 5, 6, 7])
                self.assertEqual(instructions, [
                    "test-instruction-1", 
                    "test-instruction-1",  
                    "test-instruction-2", 
                    "dummy-instruction", 
                    "dummy-instruction", 
                    "dummy-instruction"
                ])
                self.assertEqual(labels, [batch['action'][b][i] for b in [0, 1, 3, 5, 6, 7]])
                self.assertEqual(output_types, [list, list, list, list, list, list])
                expected_cur_inputs = []
                for b in [0, 1, 3, 5, 6, 7]:
                    expected_cur_inputs.append([])
                    if len(batch['image_observation'][b][i].shape) == 3:
                        expected_cur_inputs[-1].append(('image_observation', batch['image_observation'][b][i]))
                    else:
                        expected_cur_inputs[-1] += [('image_observation', image) for image in batch['image_observation'][b][i]]
                    expected_cur_inputs[-1].append(('continuous_observation', batch['continuous_observation'][b][i]))

                self.assertEqual(len(cur_inputs), len(expected_cur_inputs))
                for b in range(len(expected_cur_inputs)):
                    batch_cur_inputs = cur_inputs[b]
                    batch_expected_cur_inputs = expected_cur_inputs[b]
                    self.assertEqual(len(batch_cur_inputs), len(batch_expected_cur_inputs))
                    for j in range(len(batch_expected_cur_inputs)):
                        self.assertEqual(batch_cur_inputs[j][0], batch_expected_cur_inputs[j][0])
                        self.assertTrue(np.array_equal(batch_cur_inputs[j][1], batch_expected_cur_inputs[j][1]))
            elif 10 <= i < 12:  # Batch 1, 3, 5, 6, 7
                self.assertEqual(idxs, [1, 3, 5, 6, 7])
                self.assertEqual(instructions, [
                    "test-instruction-1",  
                    "test-instruction-2", 
                    "dummy-instruction", 
                    "dummy-instruction", 
                    "dummy-instruction"
                ])
                self.assertEqual(labels, [batch['action'][b][i] for b in [1, 3, 5, 6, 7]])
                self.assertEqual(output_types, [list, list, list, list, list])
                expected_cur_inputs = []
                for b in [1, 3, 5, 6, 7]:
                    expected_cur_inputs.append([])
                    if len(batch['image_observation'][b][i].shape) == 3:
                        expected_cur_inputs[-1].append(('image_observation', batch['image_observation'][b][i]))
                    else:
                        expected_cur_inputs[-1] += [('image_observation', image) for image in batch['image_observation'][b][i]]
                    expected_cur_inputs[-1].append(('continuous_observation', batch['continuous_observation'][b][i]))

                self.assertEqual(len(cur_inputs), len(expected_cur_inputs))
                for b in range(len(expected_cur_inputs)):
                    batch_cur_inputs = cur_inputs[b]
                    batch_expected_cur_inputs = expected_cur_inputs[b]
                    self.assertEqual(len(batch_cur_inputs), len(batch_expected_cur_inputs))
                    for j in range(len(batch_expected_cur_inputs)):
                        self.assertEqual(batch_cur_inputs[j][0], batch_expected_cur_inputs[j][0])
                        self.assertTrue(np.array_equal(batch_cur_inputs[j][1], batch_expected_cur_inputs[j][1]))
            elif 12 <= i < 21:  # Batch 1, 3, 7
                self.assertEqual(idxs, [1, 3, 7])
                self.assertEqual(instructions, [
                    "test-instruction-1",  
                    "test-instruction-2", 
                    "dummy-instruction"
                ])
                self.assertEqual(labels, [batch['action'][b][i] for b in [1, 3, 7]])
                self.assertEqual(output_types, [list, list, list])
                expected_cur_inputs = []
                for b in [1, 3, 7]:
                    expected_cur_inputs.append([])
                    if len(batch['image_observation'][b][i].shape) == 3:
                        expected_cur_inputs[-1].append(('image_observation', batch['image_observation'][b][i]))
                    else:
                        expected_cur_inputs[-1] += [('image_observation', image) for image in batch['image_observation'][b][i]]
                    expected_cur_inputs[-1].append(('continuous_observation', batch['continuous_observation'][b][i]))

                self.assertEqual(len(cur_inputs), len(expected_cur_inputs))
                for b in range(len(expected_cur_inputs)):
                    batch_cur_inputs = cur_inputs[b]
                    batch_expected_cur_inputs = expected_cur_inputs[b]
                    self.assertEqual(len(batch_cur_inputs), len(batch_expected_cur_inputs))
                    for j in range(len(batch_expected_cur_inputs)):
                        self.assertEqual(batch_cur_inputs[j][0], batch_expected_cur_inputs[j][0])
                        self.assertTrue(np.array_equal(batch_cur_inputs[j][1], batch_expected_cur_inputs[j][1]))
            elif 21 <= i < 24:  # Batch 3, 7
                self.assertEqual(idxs, [3, 7])
                self.assertEqual(instructions, [  
                    "test-instruction-2", 
                    "dummy-instruction"
                ])
                self.assertEqual(labels, [batch['action'][b][i] for b in [3, 7]])
                self.assertEqual(output_types, [list, list])
                expected_cur_inputs = []
                for b in [3, 7]:
                    expected_cur_inputs.append([])
                    if len(batch['image_observation'][b][i].shape) == 3:
                        expected_cur_inputs[-1].append(('image_observation', batch['image_observation'][b][i]))
                    else:
                        expected_cur_inputs[-1] += [('image_observation', image) for image in batch['image_observation'][b][i]]
                    expected_cur_inputs[-1].append(('continuous_observation', batch['continuous_observation'][b][i]))

                self.assertEqual(len(cur_inputs), len(expected_cur_inputs))
                for b in range(len(expected_cur_inputs)):
                    batch_cur_inputs = cur_inputs[b]
                    batch_expected_cur_inputs = expected_cur_inputs[b]
                    self.assertEqual(len(batch_cur_inputs), len(batch_expected_cur_inputs))
                    for j in range(len(batch_expected_cur_inputs)):
                        self.assertEqual(batch_cur_inputs[j][0], batch_expected_cur_inputs[j][0])
                        self.assertTrue(np.array_equal(batch_cur_inputs[j][1], batch_expected_cur_inputs[j][1]))
            else:  # Batch 3
                self.assertEqual(idxs, [3])
                self.assertEqual(instructions, [  
                    "test-instruction-2"
                ])
                self.assertEqual(labels, [batch['action'][b][i] for b in [3]])
                self.assertEqual(output_types, [list])
                expected_cur_inputs = []
                for b in [3]:
                    expected_cur_inputs.append([])
                    if len(batch['image_observation'][b][i].shape) == 3:
                        expected_cur_inputs[-1].append(('image_observation', batch['image_observation'][b][i]))
                    else:
                        expected_cur_inputs[-1] += [('image_observation', image) for image in batch['image_observation'][b][i]]
                    expected_cur_inputs[-1].append(('continuous_observation', batch['continuous_observation'][b][i]))

                self.assertEqual(len(cur_inputs), len(expected_cur_inputs))
                for b in range(len(expected_cur_inputs)):
                    batch_cur_inputs = cur_inputs[b]
                    batch_expected_cur_inputs = expected_cur_inputs[b]
                    self.assertEqual(len(batch_cur_inputs), len(batch_expected_cur_inputs))
                    for j in range(len(batch_expected_cur_inputs)):
                        self.assertEqual(batch_cur_inputs[j][0], batch_expected_cur_inputs[j][0])
                        self.assertTrue(np.array_equal(batch_cur_inputs[j][1], batch_expected_cur_inputs[j][1]))
        self.assertEqual(i, 31)  # Stopping check.


if __name__=="__main__":
    unittest.main()
