import random
import unittest
import tensorflow as tf
import torch
import numpy as np
import os
import time
import tensorflow_datasets as tfds
class TestTorchToTFDS(unittest.TestCase):
    
    def setUp(self):
        start_time = time.time()
        # Load a sample PyTorch dataset
        self.torch_dataset = torch.load("../../vd4rl/main_cheetah_run_expert_64px0.pt")
        print(f'Time taken for torch dataset load: {time.time() - start_time} seconds')
        
        # Load corresponding TFDS dataset
        start_time = time.time()
        self.tfds_dataset = tf.data.Dataset.load('../vd4rl_translated/vd4rl/main_cheetah_run_expert_64px0')
        print(f'Time taken for tfds dataset load: {time.time() - start_time} seconds')
    
    def test_dataset_sizes_match(self):
        """Test that datasets have same number of examples"""
        start_time = time.time()
        torch_lens = 0
        tf_lens = 0
        
        # Get lengths from torch dataset
        for key, value in self.torch_dataset.items():
            if isinstance(value, torch.Tensor):
                torch_lens = value.size()[0]
                break
                # Because the dataset is a dict of a list containing all elements together
        
        # Get lengths from tfds dataset
        count=0
        for ele in self.tfds_dataset:
            count+=1
        tf_lens = count

        self.assertGreater(torch_lens, 0)
        self.assertGreater(tf_lens, 0)
        self.assertEqual(torch_lens, tf_lens)
        print(f'Time taken for dataset size test: {time.time() - start_time} seconds')

    def test_feature_names_match(self):
        """Test that feature names are preserved"""
        start_time = time.time()
        
        # Get torch feature names, handling nested dicts
        torch_features = set()
        tfds_features = set()
        for key, value in self.torch_dataset.items():
            #Check if the value is a torch tensordict
            if hasattr(value, 'items'):
                for k in value.keys():
                    torch_features.add(f"{key}/{k}")
            torch_features.add(key)
                
        for ele in self.tfds_dataset:
            for key, value in ele.items():
                tfds_features.add(key)
                if isinstance(value, dict):
                    for k in value.keys():
                        tfds_features.add(f"{key}/{k}")
            break

        print(torch_features)
        print(tfds_features)
        self.assertEqual(torch_features, tfds_features)
        print(f'Time taken for feature names test: {time.time() - start_time} seconds')

    def test_data_values_match(self):
        """Test that actual data values are preserved"""
        start_time = time.time()
        
        for idx, ele in enumerate(self.tfds_dataset):
            for key, value in ele.items():

                if isinstance(value, dict):
                    for k,v in value.items():
                        pt_example = self.torch_dataset[key][k][idx]
                        tfds_example = v

                        if isinstance(pt_example, torch.Tensor):
                                pt_example = pt_example.numpy()
                        if isinstance(tfds_example, tf.Tensor):
                            tfds_example = tfds_example.numpy()

                        if isinstance(pt_example, (np.ndarray, list)):
                            np.testing.assert_array_equal(pt_example, tfds_example)
                        else:
                            if hasattr(pt_example, '__class__') and pt_example.__class__.__name__ == 'TensorDict':
                                # Compare TensorDict contents instead of the object itself
                                pt_dict = pt_example.to_dict()
                                self.assertDictEqual(pt_dict, tfds_example)
                            else:
                                self.assertEqual(pt_example, tfds_example)

                else:
                    pt_example = self.torch_dataset[key][idx]
                    tfds_example = value

                    if isinstance(pt_example, torch.Tensor):
                        pt_example = pt_example.numpy()
                    if isinstance(tfds_example, tf.Tensor):
                        tfds_example = tfds_example.numpy()

                    if isinstance(pt_example, (np.ndarray, list)):
                        np.testing.assert_array_equal(pt_example, tfds_example)
                    else:
                        if hasattr(pt_example, '__class__') and pt_example.__class__.__name__ == 'TensorDict':
                            # Compare TensorDict contents instead of the object itself
                            pt_dict = pt_example.to_dict()
                            self.assertDictEqual(pt_dict, tfds_example)
                        else:
                            self.assertEqual(pt_example, tfds_example)
    
        print(f'Time taken for data values test: {time.time() - start_time} seconds')

    def test_data_types_match(self):
        """Test that data types are preserved"""
        start_time = time.time()
        
        # Get first tfds element for comparison
        tf_element = next(iter(self.tfds_dataset))
        
        for key, torch_value in self.torch_dataset.items():
            if isinstance(torch_value, torch.Tensor):
                torch_dtype = torch_value.dtype
                tfds_dtype = tf_element[key].dtype
                
                # Map PyTorch dtypes to TF dtypes
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
                    
                self.assertEqual(torch_dtype, tfds_dtype)
                
            elif isinstance(torch_value, dict) or ((hasattr(torch_value, '__class__') and torch_value.__class__.__name__ == 'TensorDict')):
                if hasattr(torch_value, '__class__') and torch_value.__class__.__name__ == 'TensorDict':
                    torch_value = torch_value.to_dict()
                for k, v in torch_value.items():
                    if isinstance(v, torch.Tensor):
                        torch_dtype = v.dtype
                        tfds_dtype = tf_element[key][k].dtype
                        
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
                    
                self.assertEqual(torch_dtype, tfds_dtype)
            
            break
                        
        print(f'Time taken for data types test: {time.time() - start_time} seconds')

if __name__ == '__main__':
    unittest.main()
