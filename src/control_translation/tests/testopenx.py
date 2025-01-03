import random
import unittest
import tensorflow as tf
import torch
import numpy as np
import os
import time

class TestOpenXToTFDS(unittest.TestCase):
    
    def setUp(self):
        start_time = time.time()
        # Load a sample PyTorch dataset
        self.torch_dataset = torch.load("../../usc_cloth_sim_converted_externally_to_rlds/shard_0") # Adjust path as needed
        #print(f'Time taken for torch dataset load: {time.time() - start_time} seconds')
        
        # Load corresponding TFDS dataset
        self.tfds_dataset = tf.data.Dataset.load('../openx_translated/usc_cloth_sim_converted_externally_to_rlds/shard_0')
        #print(f'Time taken for torch and tfds dataset load: {time.time() - start_time} seconds')

    def test_dataset_sizes_match(self):
        """Test that datasets have same number of examples"""
        start_time = time.time()
        torch_lens = 0
        tf_lens = 0
        
        # Get lengths from torch dataset
        for episode in self.torch_dataset.items():
            torch_lens += 1
        
        # Get lengths from tfds dataset
        for episode in self.tfds_dataset:
            tf_lens += 1

    
        self.assertGreater(torch_lens, 0)
        self.assertGreater(tf_lens, 0)
        #print(torch_lens)
        #print(tf_lens)
        self.assertEqual(torch_lens, tf_lens)
        #print(f'Time taken for dataset size test: {time.time() - start_time} seconds')

    def test_feature_names_match(self):
        """Test that feature names are preserved"""
        start_time = time.time()
        
        # Get torch feature names, handling nested dicts
        torch_features = set()
        tfds_features = set()
        #Keys are the timestep number, so step into the next level of the nested dict
        for key, value in self.torch_dataset.items():
            for k,v in value.items():
                torch_features.add(f"{k}")
                if hasattr(v, 'items'):
                    for k2,v2 in v.items():
                        torch_features.add(f"{k}/{k2}")
                else:
                    torch_features.add(f"{k}")
            break
        for ele in self.tfds_dataset:
            for key, value in ele.items():
                tfds_features.add(key)
                if isinstance(value, dict):
                    for k in value.keys():
                        tfds_features.add(f"{key}/{k}")
            break
        #print(torch_features)
        #print(tfds_features)
        self.assertEqual(torch_features, tfds_features)
        #print(f'Time taken for feature names test: {time.time() - start_time} seconds')

    def test_data_values_match(self):
        """Test that actual data values are preserved"""
        start_time = time.time()
        
        for idx, tf_element in enumerate(self.tfds_dataset):
            for key, value in tf_element.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        pt_example = self.torch_dataset[idx][key][k]
                        tfds_example = v

                        if isinstance(pt_example, torch.Tensor):
                            pt_example = pt_example.numpy()
                        if isinstance(tfds_example, tf.Tensor):
                            tfds_example = tfds_example.numpy()

                        #print(pt_example)
                        #print(tfds_example)

                        if isinstance(pt_example, (np.ndarray, list)):
                            np.testing.assert_array_equal(pt_example, tfds_example)
                        else:
                            self.assertEqual(pt_example, tfds_example)
                else:
                    pt_example = self.torch_dataset[idx][key]
                    tfds_example = value

                    if isinstance(pt_example, torch.Tensor):
                        pt_example = pt_example.numpy()
                    if isinstance(tfds_example, tf.Tensor):
                        tfds_example = tfds_example.numpy()

                    #print(pt_example)
                    #print(tfds_example)

                    if isinstance(pt_example, (np.ndarray, list)):
                        np.testing.assert_array_equal(pt_example, tfds_example)
                    else:
                        self.assertEqual(pt_example, tfds_example)
        
        #print(f'Time taken for data values test: {time.time() - start_time} seconds')

    def test_data_types_match(self):
        """Test that data types are preserved"""
        start_time = time.time()
        
        # Get first tfds element for comparison
        tf_element = next(iter(self.tfds_dataset))
        
        for timestep in self.torch_dataset:
            for key, torch_value in self.torch_dataset[timestep].items():              
                
                if isinstance(tf_element[key], tf.Tensor):
                    tfds_dtype = tf_element[key].dtype

                if isinstance(torch_value, torch.Tensor) or isinstance(torch_value, np.ndarray):
                    # Handle nested ndarrays by recursively getting to innermost element
                    while isinstance(torch_value[0], (np.ndarray, torch.Tensor)):
                        torch_value = torch_value[0]
                    torch_dtype = type(torch_value[0])
                


                elif isinstance(torch_value, dict):
                    for k, v in torch_value.items():
                        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) or isinstance(v, list):
                            # Handle nested ndarrays by recursively getting to innermost element
                            while isinstance(v[0], (np.ndarray, torch.Tensor)):
                                v = v[0]
                            torch_dtype = type(v[0])
                            tfds_dtype = tf_element[key][k].dtype
                            

                            
                        else:
                            torch_dtype = type(v)
                            tfds_dtype = tf.convert_to_tensor(tf_element[key][k]).dtype

                            if torch_dtype == torch.float32:
                                torch_dtype = tf.float32
                            elif torch_dtype == torch.float64:
                                torch_dtype = tf.float64
                            elif torch_dtype == torch.int32:
                                torch_dtype = tf.int32
                            elif torch_dtype == torch.int64:
                                torch_dtype = tf.int64
                            elif torch_dtype == torch.bool:
                                torch_dtype = tf.bool
                            elif tfds_dtype == tf.string:
                                torch_dtype = type(v.decode('utf-8'))
                                tfds_dtype = type(tf_element[key][k].numpy().decode('utf-8'))
                        
                
                else:
                    torch_dtype = type(torch_value)
                    tfds_dtype = tf.convert_to_tensor(tf_element[key]).dtype

                    if torch_dtype == torch.float32:
                        torch_dtype = tf.float32
                    elif torch_dtype == torch.float64:
                        torch_dtype = tf.float64
                    elif torch_dtype == torch.int32:
                        torch_dtype = tf.int32
                    elif torch_dtype == torch.int64:
                        torch_dtype = tf.int64
                    elif torch_dtype == torch.bool:
                        torch_dtype = tf.bool
                    elif tfds_dtype == tf.string:
                        torch_dtype = type(torch_value.decode('utf-8'))
                        tfds_dtype = type(tf_element[key].numpy().decode('utf-8'))
                        
                #print(torch_value)
                #print(torch_dtype)
                #print(tf_element[key])
                #print(tfds_dtype)
                self.assertEqual(torch_dtype, tfds_dtype)

            break

        #print(f'Time taken for data types test: {time.time() - start_time} seconds')

if __name__ == '__main__':
    unittest.main()
