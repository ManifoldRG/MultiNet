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
        self.tfds_dataset = tf.data.Dataset.load('../usc_cloth_sim_converted_externally_to_rlds/shard_0')
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
        print(torch_lens)
        print(tf_lens)
        self.assertEqual(torch_lens, tf_lens)
        #print(f'Time taken for dataset size test: {time.time() - start_time} seconds')

    def test_feature_names_match(self):
        """Test that feature names are preserved"""
        start_time = time.time()
        
        # Get torch feature names, handling nested dicts
        torch_features = set(self.torch_dataset[0].keys())        
        tfds_features = set(next(iter(self.tfds_dataset)).keys())
        print(torch_features)
        print(tfds_features)
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

                        print(pt_example)
                        print(tfds_example)

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

                    print(pt_example)
                    print(tfds_example)

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
                
                if isinstance(torch_value, torch.Tensor) or isinstance(torch_value, np.ndarray):
                    torch_dtype = type(torch_value)
                    tfds_dtype = tf_element[key].dtype
                    
                    if torch_dtype == torch.Tensor:
                        torch_dtype = type(torch_value.numpy())
                    
                    tfds_dtype = type(tf_element[key].numpy())
                    
                    print(key)
                    print(torch_dtype)
                    print(tfds_dtype)
                    self.assertEqual(torch_dtype, tfds_dtype)
                    
                elif isinstance(torch_value, dict):
                    for k, v in torch_value.items():
                        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) or isinstance(v, list):
                            torch_dtype = type(v)
                            tfds_dtype = tf_element[key][k].dtype
                            
                            if torch_dtype == torch.Tensor or torch_dtype==np.ndarray:
                                tfds_dtype = type(tf_element[key][k].numpy())
                                if torch_dtype == torch.Tensor:
                                    torch_dtype = type(v.numpy())

                            
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
                        
                        print(key)
                        print(k)
                        print(torch_dtype)
                        print(tfds_dtype)
                        self.assertEqual(torch_dtype, tfds_dtype)
                
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
                        

                    print(key)
                    print(torch_dtype)
                    print(tfds_dtype)
                    self.assertEqual(torch_dtype, tfds_dtype)

            break

        #print(f'Time taken for data types test: {time.time() - start_time} seconds')

if __name__ == '__main__':
    unittest.main()
