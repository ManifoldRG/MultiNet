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
        self.rlu_dataset_1 = tf.data.TFRecordDataset('../../dmlab/explore_object_rewards_few/training_0/tfrecord-00000-of-00500', compression_type='GZIP')
        self.rlu_dataset_2 = tf.data.TFRecordDataset('../../dmlab/explore_object_rewards_few/training_0/tfrecord-00002-of-00500', compression_type='GZIP')
        self.rlu_dataset_3 = tf.data.TFRecordDataset('../../dmlab/explore_object_rewards_few/training_0/tfrecord-00003-of-00500', compression_type='GZIP')
        print(f'Time taken for RLU dataset load: {time.time() - start_time} seconds')

        # Load translated TFDS dataset
        start_time = time.time()
        self.tfds_dataset_1 = tf.data.Dataset.load('../dm_lab_rlu_translated/dmlab/explore_object_rewards_few/training_0/tfrecord-00000-of-00500')
        self.tfds_dataset_2 = tf.data.Dataset.load('../dm_lab_rlu_translated/dmlab/explore_object_rewards_few/training_0/tfrecord-00002-of-00500')
        self.tfds_dataset_3 = tf.data.Dataset.load('../dm_lab_rlu_translated/dmlab/explore_object_rewards_few/training_0/tfrecord-00003-of-00500')
        print(f'Time taken for translated dataset load: {time.time() - start_time} seconds')

    def test_dataset_sizes_match(self):
        """Test that datasets have same number of examples"""
        start_time = time.time()
        
        rlu_count = sum(1 for _ in self.rlu_dataset_1)
        tfds_count = sum(1 for _ in self.tfds_dataset_1)

        #print(rlu_count)
        #print(tfds_count)

        self.assertGreater(rlu_count, 0)
        self.assertGreater(tfds_count, 0) 
        self.assertEqual(rlu_count, tfds_count)
        
        print(f'Time taken for dataset size test: {time.time() - start_time} seconds')

    def test_feature_names_match(self):
        """Test that feature names are preserved"""
        start_time = time.time()
        
        # Get RLU feature names
        rlu_features = set()
        for raw_record in self.rlu_dataset_1:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            for key in example.features.feature:
                rlu_features.add(key)
            break

        # Get translated feature names
        tfds_features = set()
        for element in self.tfds_dataset_1:
            for key in element.keys():
                tfds_features.add(key)
            break

        print(rlu_features)
        print(tfds_features)
        self.assertEqual(rlu_features, tfds_features)
        print(f'Time taken for feature names test: {time.time() - start_time} seconds')

    def test_data_values_match(self):
        """Test that data values are preserved"""
        start_time = time.time()

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
                        
                        rlu_array = rlu_val.numpy() if isinstance(rlu_val, tf.Tensor) else rlu_val
                        
                        # Convert RaggedTensors to lists then numpy arrays for comparison
                        if isinstance(tfds_val, tf.RaggedTensor):
                            tfds_array = np.array(tfds_val.to_list())
                        else:
                            tfds_array = tfds_val.numpy()

                        np.testing.assert_array_equal(
                            rlu_array,
                            tfds_array,
                            err_msg=f"Value mismatch for {key}[{i}]"
                        )
                    else:
                        self.assertEqual(rlu_val, tfds_val,
                                    f"Value mismatch for {key}[{i}]: RLU={rlu_val}, TFDS={tfds_val}")


        #Compare values between 1st tfrecord files

        # Extract RLU values
        rlu_values = defaultdict(list)
        for raw_record in self.rlu_dataset_1:

            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            for key, feature in example.features.feature.items():
                if feature.HasField('int64_list'):
                    val = tf.convert_to_tensor(feature.int64_list.value)
                elif feature.HasField('float_list'):
                    val = tf.convert_to_tensor(feature.float_list.value)
                elif feature.HasField('bytes_list'):
                    val = []
                    for step in feature.bytes_list.value:
                        try:
                            val.append(tf.image.decode_jpeg(step, channels=3))
                        except:
                            val.append(tf.io.decode_raw(step, tf.uint8))
                    val = tf.convert_to_tensor(val)
                else:
                    raise ValueError(f"Unsupported feature type: {key}")
            
                rlu_values[key].append(val)

        # Extract TFDS values  
        tfds_values = defaultdict(list)
        for ele in self.tfds_dataset_1:
            for key, value in ele.items():
                tfds_values[key].append(value)

        test_values_match(rlu_values, tfds_values)

        #Compare values between 2nd tfrecord files
        rlu_values = defaultdict(list)
        for raw_record in self.rlu_dataset_2:

            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            for key, feature in example.features.feature.items():
                if feature.HasField('int64_list'):
                    val = tf.convert_to_tensor(feature.int64_list.value)
                elif feature.HasField('float_list'):
                    val = tf.convert_to_tensor(feature.float_list.value)
                elif feature.HasField('bytes_list'):
                    val = []
                    for step in feature.bytes_list.value:
                        try:
                            val.append(tf.image.decode_jpeg(step, channels=3))
                        except:
                            val.append(tf.io.decode_raw(step, tf.uint8))
                    val = tf.convert_to_tensor(val)
                else:
                    raise ValueError(f"Unsupported feature type: {key}")
            
                rlu_values[key].append(val)

        # Extract TFDS values  
        tfds_values = defaultdict(list)
        for ele in self.tfds_dataset_2:
            for key, value in ele.items():
                tfds_values[key].append(value)

        test_values_match(rlu_values, tfds_values)

        #Compare values between 3rd tfrecord files
        rlu_values = defaultdict(list)
        for raw_record in self.rlu_dataset_3:

            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            for key, feature in example.features.feature.items():
                if feature.HasField('int64_list'):
                    val = tf.convert_to_tensor(feature.int64_list.value)
                elif feature.HasField('float_list'):
                    val = tf.convert_to_tensor(feature.float_list.value)
                elif feature.HasField('bytes_list'):
                    val = []
                    for step in feature.bytes_list.value:
                        try:
                            val.append(tf.image.decode_jpeg(step, channels=3))
                        except:
                            val.append(tf.io.decode_raw(step, tf.uint8))
                    val = tf.convert_to_tensor(val)
                else:
                    raise ValueError(f"Unsupported feature type: {key}")
            
                rlu_values[key].append(val)

        # Extract TFDS values  
        tfds_values = defaultdict(list)
        for ele in self.tfds_dataset_3:
            for key, value in ele.items():
                tfds_values[key].append(value)

        test_values_match(rlu_values, tfds_values)



        print(f'Time taken for data values test: {time.time() - start_time} seconds')

    def test_data_types_match(self):
        """Test that data types are preserved"""
        start_time = time.time()

        # Get first elements
        rlu_first = next(iter(self.rlu_dataset_1))
        tfds_first = next(iter(self.tfds_dataset_1))

        example = tf.train.Example()
        example.ParseFromString(rlu_first.numpy())
        
        rlu_dtypes = {}
        for key, feature in example.features.feature.items():
            if feature.HasField('int64_list'):
                rlu_dtypes[key] = tf.int64
            elif feature.HasField('float_list'):
                rlu_dtypes[key] = tf.float32
            elif feature.HasField('bytes_list'):
                rlu_dtypes[key] = tf.uint8
            else:
                rlu_dtypes[key] = None

        # Get datatypes from translated dataset
        tfds_dtypes = {}
        for key, value in tfds_first.items():
            if value.dtype == tf.int32:
                tfds_dtypes[key] = tf.int64
            else:
                tfds_dtypes[key] = value.dtype
        
        #print(rlu_dtypes)
        #print(tfds_dtypes)
        self.assertEqual(rlu_dtypes, tfds_dtypes)
        print(f'Time taken for data types test: {time.time() - start_time} seconds')

if __name__ == '__main__':
    unittest.main()

