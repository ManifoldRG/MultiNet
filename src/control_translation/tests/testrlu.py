from collections import defaultdict
import random
import unittest
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import time

class TestRLUTranslation(unittest.TestCase):
    
    def setUp(self):
        start_time = time.time()
        # Load a sample RLU dataset
        self.rlu_dataset = tf.data.TFRecordDataset('../../dmlab/explore_object_rewards_few/training_0/tfrecord-00000-of-00500', compression_type='GZIP')
        print(f'Time taken for RLU dataset load: {time.time() - start_time} seconds')
        
        # Load sample episodes of translated TFDS dataset
        start_time = time.time()
        self.tfds_dataset_ep_1 = tf.data.Dataset.load('../dm_lab_rlu_translated/dmlab/explore_object_rewards_few/training_0/tfrecord-00000-of-00500/translated_episode_1')
        self.tfds_dataset_ep_last = tf.data.Dataset.load('../dm_lab_rlu_translated/dmlab/explore_object_rewards_few/training_0/tfrecord-00000-of-00500/translated_episode_181')
        self.tfds_dataset_ep_53 = tf.data.Dataset.load('../dm_lab_rlu_translated/dmlab/explore_object_rewards_few/training_0/tfrecord-00000-of-00500/translated_episode_53')
        print(f'Time taken for translated dataset load: {time.time() - start_time} seconds')

    def test_dataset_sizes_match(self):
        """Test that datasets have same number of examples"""
        start_time = time.time()
        rlu_lens = sum(1 for _ in self.rlu_dataset)
        
        # Count the number of translated episode files
        translated_dir = '../dm_lab_rlu_translated/dmlab/explore_object_rewards_few/training_0/tfrecord-00000-of-00500/'
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
        for raw_record in self.rlu_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            for key in example.features.feature:
                rlu_features.add(key)
            break

        # Get translated feature names from first episode
        tfds_features = set()
        for episode in self.tfds_dataset_ep_1:
            tfds_features = set(episode.keys())
            break

        self.assertEqual(rlu_features, tfds_features)
        print(f'Time taken for feature names test: {time.time() - start_time} seconds')

    def test_data_values_match(self):
        """Test that actual data values are preserved"""
        start_time = time.time()

        # First episode test
        rlu_count = 0
        for raw_record in self.rlu_dataset:
            if rlu_count == 0:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                for i, ele in enumerate(self.tfds_dataset_ep_1):
                    for key, feature in example.features.feature.items():
                        if feature.HasField('int64_list'):
                            rlu_val = tf.convert_to_tensor(feature.int64_list.value)
                        elif feature.HasField('float_list'):
                            rlu_val = tf.convert_to_tensor(feature.float_list.value)
                        elif feature.HasField('bytes_list'):
                            rlu_val = []
                            for step in feature.bytes_list.value:
                                try:
                                    rlu_val.append(tf.image.decode_jpeg(step, channels=3))
                                except:
                                    rlu_val.append(tf.io.decode_raw(step, tf.uint8))
                            rlu_val = tf.convert_to_tensor(rlu_val)

                        tfds_val = ele[key]

                        #print(rlu_val)
                        #print(tfds_val)

                        if isinstance(rlu_val, (tf.Tensor, tf.RaggedTensor)):
                            rlu_val = rlu_val.numpy()
                        if isinstance(tfds_val, (tf.Tensor, tf.RaggedTensor)):
                            tfds_val = tfds_val.numpy()

                        if isinstance(rlu_val, (np.ndarray, list)):
                            np.testing.assert_array_equal(rlu_val, tfds_val)
                        else:
                            self.assertEqual(rlu_val, tfds_val)

                break
            rlu_count += 1

        print(f'Time taken for first episode test: {time.time() - start_time} seconds')

        # Last episode test
        start_time = time.time()
        rlu_count = 0
        last_record = None
        for raw_record in self.rlu_dataset:
            last_record = raw_record
            rlu_count += 1

        example = tf.train.Example()
        example.ParseFromString(last_record.numpy())

        for i, ele in enumerate(self.tfds_dataset_ep_last):
            for key, feature in example.features.feature.items():
                if feature.HasField('int64_list'):
                    rlu_val = tf.convert_to_tensor(feature.int64_list.value)
                elif feature.HasField('float_list'):
                    rlu_val = tf.convert_to_tensor(feature.float_list.value)
                elif feature.HasField('bytes_list'):
                    rlu_val = []
                    for step in feature.bytes_list.value:
                        try:
                            rlu_val.append(tf.image.decode_jpeg(step, channels=3))
                        except:
                            rlu_val.append(tf.io.decode_raw(step, tf.uint8))
                    rlu_val = tf.convert_to_tensor(rlu_val)

                tfds_val = ele[key]

                #print(rlu_val)
                #print(tfds_val)

                if isinstance(rlu_val, (tf.Tensor, tf.RaggedTensor)):
                    rlu_val = rlu_val.numpy()
                if isinstance(tfds_val, (tf.Tensor, tf.RaggedTensor)):
                    tfds_val = tfds_val.numpy()

                if isinstance(rlu_val, (np.ndarray, list)):
                    np.testing.assert_array_equal(rlu_val, tfds_val)
                else:
                    self.assertEqual(rlu_val, tfds_val)

        print(f'Time taken for last episode test: {time.time() - start_time} seconds')

        # Random episode (13th) test
        start_time = time.time()
        rlu_count = 0
        for raw_record in self.rlu_dataset:
            if rlu_count == 52:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())

                for i, ele in enumerate(self.tfds_dataset_ep_53):
                    for key, feature in example.features.feature.items():
                        if feature.HasField('int64_list'):
                            rlu_val = tf.convert_to_tensor(feature.int64_list.value)
                        elif feature.HasField('float_list'):
                            rlu_val = tf.convert_to_tensor(feature.float_list.value)
                        elif feature.HasField('bytes_list'):
                            rlu_val = []
                            for step in feature.bytes_list.value:
                                try:
                                    rlu_val.append(tf.image.decode_jpeg(step, channels=3))
                                except:
                                    rlu_val.append(tf.io.decode_raw(step, tf.uint8))
                            rlu_val = tf.convert_to_tensor(rlu_val)

                        tfds_val = ele[key]

                        #print(rlu_val)
                        #print(tfds_val)

                        if isinstance(rlu_val, (tf.Tensor, tf.RaggedTensor)):
                            rlu_val = rlu_val.numpy()
                        if isinstance(tfds_val, (tf.Tensor, tf.RaggedTensor)):
                            tfds_val = tfds_val.numpy()

                        if isinstance(rlu_val, (np.ndarray, list)):
                            np.testing.assert_array_equal(rlu_val, tfds_val)
                        else:
                            self.assertEqual(rlu_val, tfds_val)

                break
            rlu_count += 1

        print(f'Time taken for random episode test: {time.time() - start_time} seconds')

    def test_data_types_match(self):
        """Test that data types are preserved"""
        start_time = time.time()

        # Get datatypes from RLU dataset
        rlu_dtypes = {}
        for raw_record in self.rlu_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            for key, feature in example.features.feature.items():
                if feature.HasField('int64_list'):
                    rlu_dtypes[key] = tf.int64
                elif feature.HasField('float_list'):
                    rlu_dtypes[key] = tf.float32
                elif feature.HasField('bytes_list'):
                    rlu_dtypes[key] = tf.uint8
            break

        # Get datatypes from translated dataset
        tfds_dtypes = {}
        for episode in self.tfds_dataset_ep_1:
            for key, value in episode.items():
                if value.dtype == tf.int32:
                    tfds_dtypes[key] = tf.int64
                else:
                    tfds_dtypes[key] = value.dtype
            break

        self.assertEqual(rlu_dtypes, tfds_dtypes)
        print(f'Time taken for data types test: {time.time() - start_time} seconds')

if __name__ == '__main__':
    unittest.main()
