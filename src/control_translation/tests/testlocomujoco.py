import random
import unittest
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

class TestLocoMujocoToTFDS(unittest.TestCase):
    
    def setUp(self):
        # Load a sample .npz file and its corresponding translated TFDS dataset
        self.npz_path = "../../loco-mujoco/loco_mujoco/datasets/humanoids/perfect/atlas_walk/perfect_expert_dataset_det.npz" # Adjust path as needed
        self.tfds_dataset = tf.data.Dataset.load('../locomujoco_translated/loco-mujoco/loco_mujoco/datasets/humanoids/perfect/atlas_walk/perfect_expert_dataset_det')
        
        # Load the .npz file
        self.npz_data = np.load(self.npz_path, allow_pickle=True)

    def test_dataset_sizes_match(self):
        """Test that datasets have same number of examples"""
        npz_lens = 0
        tf_lens = 0
        
        # Get lengths from npz data
        for key, value in self.npz_data.items():
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
        for idx, tf_element in enumerate(self.tfds_dataset):
            for key in self.npz_data.keys():
                npz_example = self.npz_data[key][idx]
                tfds_example = tf_element[key]

                if isinstance(tfds_example, tf.Tensor):
                    tfds_example = tfds_example.numpy()
                
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
            
            # Convert tensorflow types to numpy types for comparison
            if tfds_type == tf.int64:
                tfds_type = np.int64
            elif tfds_type == tf.float32:
                tfds_type = np.float32
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


            self.assertEqual(npz_type, tfds_type)

if __name__ == '__main__':
    unittest.main()
