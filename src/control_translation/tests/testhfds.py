import random
import unittest
import tensorflow as tf
import tensorflow_datasets as tfds
import datasets
import numpy as np
import os
from PIL import Image, ImageChops
from io import BytesIO
import time

class TestHFToTFDS(unittest.TestCase):
    
    def to_numpy(self, im):
        #Fast method to convert PIL image to numpy array
        im.load()
        # unpack data
        e = Image._getencoder(im.mode, 'raw', im.mode)
        e.setimage(im.im)

        # NumPy buffer for the result
        shape, typestr = Image._conv_type_shape(im)
        data = np.empty(shape, dtype=np.dtype(typestr))
        mem = data.data.cast('B', (data.data.nbytes,))

        bufsize, s, offset = 65536, 0, 0
        while not s:
            l, s, d = e.encode(bufsize)
            mem[offset:offset + len(d)] = d
            offset += len(d)
        if s < 0:
            raise RuntimeError("encoder error %d in tobytes" % s)
        return data
    
    def setUp(self):
        start_time = time.time()
        # Load a sample HuggingFace dataset from arrow file
        self.hf_dataset = datasets.load_from_disk("../../atari/atari-alien")
        print(f'Time taken for hf dataset load: {time.time() - start_time} seconds')
        
        # Load corresponding TFDS dataset
        start_time = time.time()
        self.tfds_dataset = tf.data.Dataset.load('../ale_atari_translated/atari/atari-alien/')
        print(f'Time taken for tfds dataset load: {time.time() - start_time} seconds')
    
    def test_dataset_sizes_match(self):
        """Test that datasets have same number of examples"""
        start_time = time.time()
        hf_lens = []
        tf_lens = []
        for episode in self.hf_dataset['train']:
            for key, value in episode.items():
                hf_lens.append(len(value))
        
        for episode in self.tfds_dataset:
            for key, value in episode.items():
                if isinstance(value, tf.RaggedTensor):
                    tf_lens.append(value.shape[0])
                else:
                    tf_lens.append(len(value))

        #print(hf_lens)
        #print(tf_lens)
        self.assertGreater(len(hf_lens), 0)
        self.assertGreater(len(tf_lens), 0)
        self.assertTrue(all(h == t for h, t in zip(hf_lens, tf_lens)))
        print(f'Time taken for dataset size test: {time.time() - start_time} seconds')

    def test_feature_names_match(self):
        """Test that feature names are preserved"""
        start_time = time.time()
        hf_features = set(self.hf_dataset['train'].column_names)
        tfds_features = set(self.tfds_dataset.element_spec.keys())
        self.assertEqual(hf_features, tfds_features)
        print(f'Time taken for feature names test: {time.time() - start_time} seconds')

    def test_data_values_match(self):
        """Test that actual data values are preserved"""
        # Compare first and last episodes
        #First episode
        #print('\nFirst episode test')
        start_time = time.time()
        keys_list = set(self.hf_dataset['train'].column_names)
        #print(keys_list)
        tf_element = next(iter(self.tfds_dataset))
        for i in range(min(10, len(self.hf_dataset['train']['rewards'][0]))):
            #print(i)
            for key in keys_list:

                #if key != 'image_observations':
                
                #print(key)
                
                hf_example = self.hf_dataset['train'][key][0][i]
                tfds_example = tf_element[key][i]

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
        print(f'Time taken for first episode test: {time.time() - start_time} seconds')

        start_time = time.time()
        #Last episode
        print('\nLast episode test')
        last_idx = len(self.hf_dataset['train']['rewards']) - 1
        tf_element = None
        for element in self.tfds_dataset:
            tf_element = element
        
        for i in range(min(10, len(tf_element['rewards']))):
            
            #print(i)
            for key in keys_list:
                print(key)
                
                hf_example = self.hf_dataset['train'][key][last_idx][i]
                tfds_example = tf_element[key][i]

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
                
                #print(hf_example)
                #print(tfds_example)
                    
                if isinstance(hf_example, (np.ndarray, list)):
                    np.testing.assert_array_equal(hf_example, tfds_example)
                else:
                    self.assertEqual(hf_example, tfds_example)
        print(f'Time taken for last episode test: {time.time() - start_time} seconds')

        #Random episode
        start_time = time.time()
        rand_idx = random.randint(0, len(self.hf_dataset['train']['rewards']) - 1)
        print(f'\n Random episode {rand_idx} test')

        tf_element = None
        for idx, element in enumerate(self.tfds_dataset):
            if idx == rand_idx:
                tf_element = element
                break

        for i in range(min(10, len(tf_element['rewards']))):
            
            #print(i)
            for key in keys_list:
                #print(key)
                
                hf_example = self.hf_dataset['train'][key][rand_idx][i]
                tfds_example = tf_element[key][i]

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
                
                #print(hf_example)
                #print(tfds_example)
                    
                if isinstance(hf_example, (np.ndarray, list)):
                    np.testing.assert_array_equal(hf_example, tfds_example)
                else:
                    self.assertEqual(hf_example, tfds_example)
        print(f'Time taken for random episode test: {time.time() - start_time} seconds')


    def test_data_types_match(self):
        """Test that data types are preserved"""
        start_time = time.time()
        for feature in set(self.hf_dataset['train'].column_names):
            
            hf_type = type(self.hf_dataset['train'][feature][0])
            for ele in self.tfds_dataset:
                tfds_type = type(ele[feature][0])
                break
            
            #print(hf_type)
            #print(tfds_type)
            
            # Convert HF types to TF types for comparison
            if hf_type == 'int64':
                hf_type = tf.int64
            elif hf_type == 'float32':
                hf_type = tf.float32
            elif hf_type == list:
                
                try:
                    hf_type = type(tf.convert_to_tensor(self.hf_dataset['train'][feature][0]))
                except:
                    hf_type = type(tf.convert_to_tensor(np.stack([np.array(img) for img in self.hf_dataset['train'][feature][0]])))
            elif isinstance(self.hf_dataset['train'][feature][0], tf.Tensor):
                hf_type = type(tf.RaggedTensor.from_tensor(self.hf_dataset['train'][feature][0]))
            
            self.assertEqual(hf_type, tfds_type)
        print(f'Time taken for data types test: {time.time() - start_time} seconds')

if __name__ == '__main__':
    unittest.main()
