from collections import defaultdict
import random
import unittest
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import time

class TestRLUToTFDS(unittest.TestCase):
    
    def setUp(self):
        start_time = time.time()
        # Load a sample RLU dataset
        self.rlu_dataset = tfds.load(
            'rlu_control_suite/cartpole_swingup',
            split='train',
            data_dir='../../rlu_control_suite/cartpole_swingup',
            download=True
        )
        print(f'Time taken for RLU dataset load: {time.time() - start_time} seconds')
        
        # Load first 3 episodes of translated TFDS dataset
        start_time = time.time()
        self.tfds_dataset_ep_1 = tf.data.Dataset.load('../dm_control_suite_rlu_translated/rlu_control_suite/cartpole_swingup/translated_episode_1')
        self.tfds_dataset_ep_40 = tf.data.Dataset.load('../dm_control_suite_rlu_translated/rlu_control_suite/cartpole_swingup/translated_episode_40')
        self.tfds_dataset_ep_17 = tf.data.Dataset.load('../dm_control_suite_rlu_translated/rlu_control_suite/cartpole_swingup/translated_episode_17')
        print(f'Time taken for translated dataset load: {time.time() - start_time} seconds')

    def test_dataset_sizes_match(self):
        """Test that datasets have same number of examples"""
        start_time = time.time()
        rlu_lens = 0
        tfds_lens = 0
        
        # Get lengths from RLU dataset
        for episode in self.rlu_dataset:
            rlu_lens += 1
             
        # Count the number of translated episode files
        translated_dir = '../dm_control_suite_rlu_translated/rlu_control_suite/cartpole_swingup/'
        tfds_lens = len([f for f in os.listdir(translated_dir) if f.startswith('translated_')])

        self.assertGreater(rlu_lens, 0)
        self.assertGreater(tfds_lens, 0)
        self.assertEqual(rlu_lens, tfds_lens)
        print(f'Time taken for dataset size test: {time.time() - start_time} seconds')

    def test_feature_names_match(self):
        """Test that feature names are preserved"""
        start_time = time.time()
        
        # Get RLU feature names
        rlu_features = set()
        for episode in self.rlu_dataset:
            for key in episode.keys():
                if key == 'steps':
                    # Handle steps variant dataset
                    for step in episode[key]:
                        #print(step.keys())
                        for step_key in step.keys():      

                            rlu_features.add(f'steps/{step_key}')
                        
                        break
                else:
                    rlu_features.add(key)
            break

        # Get translated feature names
        tfds_features = set()
        for episode in self.tfds_dataset_ep_1:
            for key in episode.keys():
                if key == 'steps':
                    for step_number in episode[key]:
                        #print('\nCHECK')
                        #print(episode[key][step_number].keys())
                        for step_key in episode[key][step_number].keys():
                            tfds_features.add(f'steps/{step_key}')
                        break
                    
                else:
                    tfds_features.add(key)
            break

        self.assertEqual(rlu_features, tfds_features)
        print(f'Time taken for feature names test: {time.time() - start_time} seconds')

    def test_data_values_match(self):

        def test_values_match(rlu_values, tfds_values):
        # Compare values between RLU and TFDS datasets
            for key in rlu_values.keys():
                self.assertIn(key, tfds_values, f"Key {key} missing from translated dataset")
                
                rlu_arr = rlu_values[key]
                tfds_arr = tfds_values[key]
                
                self.assertEqual(len(rlu_arr), len(tfds_arr), 
                            f"Length mismatch for {key}: RLU={len(rlu_arr)}, TFDS={len(tfds_arr)}")
                
                for i, (rlu_val, tfds_val) in enumerate(zip(rlu_arr, tfds_arr)):
                    if isinstance(rlu_val, (np.ndarray, tf.Tensor)):
                        np.testing.assert_array_equal(
                            rlu_val.numpy() if isinstance(rlu_val, tf.Tensor) else rlu_val,
                            tfds_val.numpy() if isinstance(tfds_val, tf.Tensor) else tfds_val,
                            err_msg=f"Value mismatch for {key}[{i}]"
                        )
                    else:
                        self.assertEqual(rlu_val, tfds_val,
                                    f"Value mismatch for {key}[{i}]: RLU={rlu_val}, TFDS={tfds_val}")
        """Test that actual data values are preserved"""
        start_time = time.time()
        
        # Compare first episode

        tfds_ele = next(iter(self.tfds_dataset_ep_1))
        rlu_ele = self.rlu_dataset

        #Compare values of first episode
        rlu_values = defaultdict(list)
        for episode in self.rlu_dataset:
            for key in episode.keys():
                if key == 'steps':
                    # Handle steps variant dataset
                    for step in episode[key]:
                        print(step.keys())
                        for step_key in step.keys():      
                            if isinstance(step[step_key], dict):
                                for k, v in step[step_key].items():
                                    rlu_values[f'steps/{step_key}/{k}'].append(v)
                            else:
                                rlu_values[f'steps/{step_key}'].append(step[step_key])
                else:
                    rlu_values[key].append(episode[key])
            break
            

            
        #print(rlu_values)
        
        tfds_values = defaultdict(list)
        for episode in self.tfds_dataset_ep_1:
            for key in episode.keys():
                if key == 'steps':
                    for step_number in episode[key]:
                        for step_key in episode[key][step_number].keys():
                            if isinstance(episode[key][step_number][step_key], dict):
                                for k, v in episode[key][step_number][step_key].items():
                                    tfds_values[f'steps/{step_key}/{k}'].append(v)
                            else:
                                tfds_values[f'steps/{step_key}'].append(episode[key][step_number][step_key])
                else:
                    tfds_values[key].append(episode[key])
            

        #print(tfds_values)

        test_values_match(rlu_values, tfds_values)

        #Compare values of last episode

        for episode in self.rlu_dataset:
            pass
        rlu_ele = episode
        tfds_ele = next(iter(self.tfds_dataset_ep_40))
        
        rlu_values = defaultdict(list)
        for key in rlu_ele.keys():
            if key == 'steps':
                # Handle steps variant dataset
                for step in rlu_ele[key]:
                        print(step.keys())
                        for step_key in step.keys():      
                            if isinstance(step[step_key], dict):
                                for k, v in step[step_key].items():
                                    rlu_values[f'steps/{step_key}/{k}'].append(v)
                            else:
                                rlu_values[f'steps/{step_key}'].append(step[step_key])
                else:
                    rlu_values[key].append(rlu_ele[key])
            break
            

            
        #print(rlu_values)
        
        tfds_values = defaultdict(list)

        for key in tfds_ele.keys():
            if key == 'steps':
                for step_number in tfds_ele[key]:
                    for step_key in tfds_ele[key][step_number].keys():
                        if isinstance(tfds_ele[key][step_number][step_key], dict):
                            for k, v in tfds_ele[key][step_number][step_key].items():
                                    tfds_values[f'steps/{step_key}/{k}'].append(v)
                            else:
                                tfds_values[f'steps/{step_key}'].append(tfds_ele[key][step_number][step_key])
            else:
                tfds_values[key].append(tfds_ele[key])

        test_values_match(rlu_values, tfds_values)

        #Compare values of 17th episode
        count=1
        for episode in self.rlu_dataset:
            if count == 17:
                rlu_ele = episode
                break
            count += 1

        tfds_ele = next(iter(self.tfds_dataset_ep_17))
        
        rlu_values = defaultdict(list)
        for key in rlu_ele.keys():
            if key == 'steps':
                # Handle steps variant dataset
                for step in rlu_ele[key]:
                        print(step.keys())
                        for step_key in step.keys():      
                            if isinstance(step[step_key], dict):
                                for k, v in step[step_key].items():
                                    rlu_values[f'steps/{step_key}/{k}'].append(v)
                            else:
                                rlu_values[f'steps/{step_key}'].append(step[step_key])
                else:
                    rlu_values[key].append(rlu_ele[key])
            break
            

            
        #print(rlu_values)
        
        tfds_values = defaultdict(list)

        for key in tfds_ele.keys():
            if key == 'steps':
                for step_number in tfds_ele[key]:
                    for step_key in tfds_ele[key][step_number].keys():
                        if isinstance(tfds_ele[key][step_number][step_key], dict):
                            for k, v in tfds_ele[key][step_number][step_key].items():
                                    tfds_values[f'steps/{step_key}/{k}'].append(v)
                            else:
                                tfds_values[f'steps/{step_key}'].append(tfds_ele[key][step_number][step_key])
            else:
                tfds_values[key].append(tfds_ele[key])

        test_values_match(rlu_values, tfds_values)





        print(f'Time taken for data values test: {time.time() - start_time} seconds')

    def test_data_types_match(self):
        """Test that data types are preserved"""
        start_time = time.time()
        
        # Get first elements
        rlu_first = next(iter(self.rlu_dataset))
        tfds_first = self.tfds_dataset_ep_1

        # Get datatypes of all features of origin dataset
        rlu_datatypes = {}
        for episode in self.rlu_dataset:
            for key in episode.keys():
                if key == 'steps':
                    # Handle steps variant dataset
                    for step in episode[key]:
                        print(step.keys())
                        for step_key in step.keys():      

                            if isinstance(step[step_key], dict):
                                for k, v in step[step_key].items():
                                    rlu_datatypes[f'steps/{step_key}/{k}'] = v.dtype
                            else:
                                rlu_datatypes[f'steps/{step_key}'] = step[step_key].dtype
                        
                        break
                else:
                    
                    rlu_datatypes[key] = episode[key].dtype
            break

        print(rlu_datatypes)

        # Get datatypes of all features of translated dataset
        tfds_datatypes = {}
        for episode in self.tfds_dataset_ep_1:
            for key in episode.keys():
                if key == 'steps':
                    for step_number in episode[key]:
                        for step_key in episode[key][step_number].keys():
                            if isinstance(episode[key][step_number][step_key], dict):
                                for k, v in episode[key][step_number][step_key].items():
                                    tfds_datatypes[f'steps/{step_key}/{k}'] = v.dtype
                            else:
                                tfds_datatypes[f'steps/{step_key}'] = episode[key][step_number][step_key].dtype
                        break
                    
                else:
                    tfds_datatypes[key] = episode[key].dtype
            break

        print(tfds_datatypes)

        self.assertEqual(rlu_datatypes, tfds_datatypes)
        
        print(f'Time taken for data types test: {time.time() - start_time} seconds')

if __name__ == '__main__':
    unittest.main()
