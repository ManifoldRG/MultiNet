from collections import defaultdict
import random
import unittest
import tensorflow as tf
import tensorflow_datasets as tfds
import datasets
import numpy as np
import os
from PIL import Image
import time

class TestHFToTFDSMod(unittest.TestCase):
    
    def setUp(self):
        start_time = time.time()
        # Load a sample HuggingFace dataset
        self.hf_dataset = datasets.load_from_disk("../../atari/atari-amidar/")
        print(f'Time taken for HF dataset load: {time.time() - start_time} seconds')
        
        # Load sample episodes of translated TFDS dataset
        start_time = time.time()
        self.tfds_dataset_ep_1 = tf.data.Dataset.load('../ale_atari_translated/atari/atari-amidar/test/test_episode_1')
        self.tfds_dataset_ep_last = tf.data.Dataset.load('../ale_atari_translated/atari/atari-amidar/test/test_episode_17') 
        self.tfds_dataset_ep_13 = tf.data.Dataset.load('../ale_atari_translated/atari/atari-amidar/test/test_episode_13')
        print(f'Time taken for translated dataset load: {time.time() - start_time} seconds')

    def test_dataset_sizes_match(self):
        """Test that datasets have same number of examples"""
        start_time = time.time()
        hf_lens = len(self.hf_dataset['test'])
        
        # Count the number of translated episode files
        translated_dir = '../ale_atari_translated/atari/atari-amidar/test/'
        tfds_lens = len([f for f in os.listdir(translated_dir) if 'test' in f])

        self.assertGreater(hf_lens, 0)
        self.assertGreater(tfds_lens, 0)
        self.assertEqual(hf_lens, tfds_lens)
        print(f'Time taken for dataset size test: {time.time() - start_time} seconds')

    def test_feature_names_match(self):
        """Test that feature names are preserved"""
        start_time = time.time()
        hf_features = set(self.hf_dataset['test'].column_names)
        
        # Get translated feature names from first episode
        tfds_features = set()
        for episode in self.tfds_dataset_ep_1:
            tfds_features = set(episode.keys())
            break

        self.assertEqual(hf_features, tfds_features)
        print(f'Time taken for feature names test: {time.time() - start_time} seconds')

    def test_data_values_match(self):
        
        """Test that actual data values are preserved"""
        keys_list = set(self.hf_dataset['test'].column_names)
        
        #First episode
        tf_ep = self.tfds_dataset_ep_1

        start_time = time.time()
        for i, ele in enumerate(tf_ep):
            #print(i)
            for key in keys_list:

                #if key != 'image_observations':
                
                hf_example = self.hf_dataset['test'][key][0][i]
                tfds_example = ele[key]

                if isinstance(hf_example, str):
                    tfds_example = tfds_example.numpy().decode('utf-8')
                
                if hasattr(hf_example, 'mode') and hasattr(hf_example, 'size'):  # Check if it's a PIL Image
                    hf_example = np.asarray(hf_example)
            
                if isinstance(hf_example, tf.Tensor):
                    hf_example = hf_example.numpy()
                if isinstance(tfds_example, tf.Tensor):
                    tfds_example = tfds_example.numpy()
                
                #print(hf_example)
                #print(tfds_example)
                    
                if isinstance(hf_example, (np.ndarray, list)):
                    np.testing.assert_array_equal(hf_example, tfds_example)
                else:
                    self.assertEqual(hf_example, tfds_example)
            
            #Check first 15 timesteps of the episode
            if i == 15:
                break

        print(f'Time taken for first episode test: {time.time() - start_time} seconds')

        start_time = time.time()
        #Last episode
        print('\nLast episode test')
        last_idx = len(self.hf_dataset['test']['rewards']) - 1
        tf_ep = self.tfds_dataset_ep_last


        for i, ele in enumerate(tf_ep):
            
            #print(i)
            for key in keys_list:
                #print(key)
                
                hf_example = self.hf_dataset['test'][key][last_idx][i]

                tfds_example = ele[key]

                if isinstance(hf_example, str):
                    tfds_example = tfds_example.numpy().decode('utf-8')
                
                if hasattr(hf_example, 'mode'):  # Check if it's a PIL Image
                    start_time = time.time()
                    #hf_example = self.to_numpy(hf_example)
                    hf_example = np.asarray(hf_example)
                    print(f'Time taken for numpy conversion: {time.time() - start_time} seconds')
            
                elif isinstance(hf_example, tf.Tensor):
                    hf_example = hf_example.numpy()

                if isinstance(tfds_example, tf.Tensor):
                    tfds_example = tfds_example.numpy()
                
                print(hf_example)
                print(tfds_example)
                    
                if isinstance(hf_example, (np.ndarray, list)):
                    np.testing.assert_array_equal(hf_example, tfds_example)
                else:
                    self.assertEqual(hf_example, tfds_example)
            
            if i == 15:
                break

        print(f'Time taken for last episode test: {time.time() - start_time} seconds')

        #Random episode
        start_time = time.time()
        tf_ep = self.tfds_dataset_ep_13

        for i, ele in enumerate(tf_ep):
            
            #print(i)
            for key in keys_list:
                #print(key)
                
                hf_example = self.hf_dataset['test'][key][12][i]
                tfds_example = ele[key]

                if isinstance(hf_example, str):
                    tfds_example = tfds_example.numpy().decode('utf-8')
                
                if hasattr(hf_example, 'mode'):  # Check if it's a PIL Image
                    start_time = time.time()
                    #hf_example = self.to_numpy(hf_example)
                    hf_example = np.asarray(hf_example)
                    print(f'Time taken for numpy conversion: {time.time() - start_time} seconds')
            
                elif isinstance(hf_example, tf.Tensor):
                    hf_example = hf_example.numpy()

                if isinstance(tfds_example, tf.Tensor):
                    tfds_example = tfds_example.numpy()
                
                print(hf_example)
                print(tfds_example)
                    
                if isinstance(hf_example, (np.ndarray, list)):
                    np.testing.assert_array_equal(hf_example, tfds_example)
                else:
                    self.assertEqual(hf_example, tfds_example)
            
            if i == 15:
                break

        print(f'Time taken for random episode test: {time.time() - start_time} seconds')

    def test_data_types_match(self):
        """Test that data types are preserved"""
        start_time = time.time()
        
        # Get datatypes of all features of origin dataset
        hf_datatypes = {}
        for feature in self.hf_dataset['test'].column_names:
            try:
                hf_type = self.hf_dataset['test'][feature][0].dtype
            except:
                hf_type = type(self.hf_dataset['test'][feature][0])
            
            # Convert HF types to TF types for comparison
            if hf_type == 'int64':
                hf_type = tf.int64
            elif hf_type == 'float32':
                hf_type = tf.float32
            elif hf_type == list:
                try:
                    hf_type = tf.convert_to_tensor(self.hf_dataset['test'][feature][0]).dtype
                except:
                    hf_type = tf.convert_to_tensor(np.stack([np.array(img) for img in self.hf_dataset['test'][feature][0]])).dtype
            elif isinstance(self.hf_dataset['test'][feature][0], tf.Tensor):
                hf_type = tf.RaggedTensor.from_tensor(self.hf_dataset['test'][feature][0]).dtype
            
            hf_datatypes[feature] = hf_type

        # Get datatypes of all features of translated dataset
        tfds_datatypes = {}
        for episode in self.tfds_dataset_ep_1:
            for key in episode.keys():
                tfds_datatypes[key] = episode[key].dtype
            break

        #print(hf_datatypes)
        #print(tfds_datatypes)
        self.assertEqual(hf_datatypes, tfds_datatypes)
        print(f'Time taken for data types test: {time.time() - start_time} seconds')

if __name__ == '__main__':
    unittest.main()
