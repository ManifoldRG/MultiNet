import tensorflow_datasets as tfds
import tensorflow as tf
import tqdm
import apache_beam as beam


# optionally replace the DATASET_NAMES below with the list of filtered datasets from the google sheet
DOWNLOAD_DIR = '~/ManifoldRG/MultiNet/src/eval/profiling/openvla/data'

# Create beam options
beam_options = beam.options.pipeline_options.PipelineOptions(
    runner='DirectRunner',
    direct_num_workers=2,
    direct_running_mode='multi_processing'
)

# Download and prepare the dataset
dataset = tfds.load(
    'usc_cloth_sim_converted_externally_to_rlds',
    data_dir=DOWNLOAD_DIR,
    download=True,
    batch_size=32,
    # split='train',
    # as_supervised=True,
    with_info=True,
    download_and_prepare_kwargs={
        'download_config': tfds.download.DownloadConfig(
            beam_options=beam_options
        )
    }
)

dataset = dataset.flat_map(lambda x: x)  # Flatten if needed
dataset = dataset.batch(32)