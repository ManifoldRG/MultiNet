import os
import gc
import numpy as np
from tqdm import tqdm
import tensorflow as tf

PATH = "chaser"
buffer = []
total_files = len(os.listdir(PATH))
mega_dataset = None
save_counter = 0

for path in tqdm(os.listdir(PATH), total=total_files, unit='file'):
    data = np.load(os.path.join(PATH, path), allow_pickle=True).item()
    datasets = {key: tf.data.Dataset.from_tensor_slices(data[key]) for key in data.keys()}
    tfds_dataset = tf.data.Dataset.zip(datasets)

    if mega_dataset is None:
        mega_dataset = tfds_dataset
    else:
        mega_dataset = mega_dataset.concatenate(tfds_dataset)

    save_counter += 1

    if save_counter % 1000 == 0:
        # Save the mega dataset to disk every 1000 steps
        tf.data.Dataset.save(mega_dataset, f"mega_dataset_{save_counter}")
        print(f"Mega dataset saved to disk. Step: {save_counter}")

        # Delete the mega dataset from garbage collection
        del mega_dataset
        gc.collect()

        # Reset the mega dataset to None
        mega_dataset = None

# Save the final mega dataset to disk
if mega_dataset is not None:
    tf.data.experimental.save(mega_dataset, "mega_dataset_final")
    print("Final mega dataset saved to disk.")