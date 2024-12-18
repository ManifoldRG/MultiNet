import random
import unittest
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

class TestProcgenToTFDS(unittest.TestCase):
    
    def setUp(self):
        # Load a sample .npz file and its corresponding translated TFDS dataset
        
        self.npz_path = "../../bigfish/20230329T100243_5618_145_68_0.00.npy" # Adjust path as needed
        self.tfds_dataset = tf.data.Dataset.load('../procgen_translated/bigfish/20230329T100243_5618_145_68_0.00')
        
        # Load the .npz file
        self.procgen_dict = np.load(self.npz_path, allow_pickle=True).item()
        self.npz_data = self.procgen_dict

    def test_dataset_sizes_match(self):
        """Test that datasets have same number of examples"""
        npz_lens = 0
        tf_lens = 0
        
        # Get lengths from npz data
        for key, value in self.npz_data.items():
            if key=='observations':
                npz_lens = len(value)
                break
            
        # Get lengths from tfds data
        size_count = 0
        for episode in self.tfds_dataset:
            size_count+=1
        
        tf_lens = size_count

        self.assertGreater(npz_lens, 0)
        self.assertGreater(tf_lens, 0)
        self.assertEqual(npz_lens, tf_lens)

    def test_feature_names_match(self):
        """Test that feature names are preserved"""
        npz_features = set(self.npz_data.keys())
        tfds_features = set(next(iter(self.tfds_dataset)).keys())
        self.assertEqual(npz_features, tfds_features)

    def test_data_values_match(self):
        """Test that actual data values are preserved"""
        
        #There is one extra observation per episode when compared to actions, rewards, etc. This is handled during translation by padding, but it is still uneve (num_obs = num_actions+1) in the original dataset
        for idx, tf_element in enumerate(self.tfds_dataset):
            for key in self.npz_data.keys(): 

                if key=='observations':   
                    npz_example = self.npz_data[key][idx]
                    tfds_example = tf_element[key]

                    if isinstance(tfds_example, tf.Tensor):
                        tfds_example = tfds_example.numpy()
                    
                    print(npz_example)
                    print(tfds_example)
                    
                    if isinstance(npz_example, (np.ndarray, list)):
                        np.testing.assert_array_equal(npz_example, tfds_example)
                    else:
                        self.assertEqual(npz_example, tfds_example)
                
                elif idx!=0 and key!='observations':
                    npz_example = self.npz_data[key][idx-1]
                    tfds_example = tf_element[key]

                    if isinstance(tfds_example, tf.Tensor):
                        tfds_example = tfds_example.numpy()
                    
                    print(npz_example)
                    print(tfds_example)
                    
                    if isinstance(npz_example, (np.ndarray, list)):
                        np.testing.assert_array_equal(npz_example, tfds_example)
                    else:
                        self.assertEqual(npz_example, tfds_example)


                    

    def test_data_types_match(self):
        """Test that data types are preserved"""
        
        
        for key in self.npz_data.keys():
            tf_element = next(iter(self.tfds_dataset))[key]
            npz_type = type(self.npz_data[key][0])
            tfds_type = type(tf_element)
            
            # Convert numpy types to TF types for comparison
            if tfds_type == tf.int64:
                tfds_type = np.int64
            elif tfds_type == tf.float32:
                tfds_type = np.float32
            elif tfds_type == tf.Tensor:
                tfds_type = np.ndarray
            elif tfds_type == tf.bool:
                tfds_type = np.bool
            elif tfds_type == tf.float64:
                tfds_type = np.float64
            elif tfds_type == tf.int32:
                tfds_type = np.int32
            elif tfds_type == tf.Tensor:
                tfds_type = type(tf_element.numpy())
            elif str(tfds_type) == "<class 'tensorflow.python.framework.ops.EagerTensor'>":
                tfds_type = type(tf_element.numpy())
                    
            print(npz_type)
            print(tfds_type)
            self.assertEqual(npz_type, tfds_type)

if __name__ == '__main__':
    unittest.main()
