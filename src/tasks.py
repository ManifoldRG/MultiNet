from datasets import load_dataset, concatenate_datasets, Dataset, IterableDataset
from tqdm import tqdm
from functools import partial

import utils


def get_control_datasets(datasets_config, split):
    """
    Generate datasets based around control.
    """


def get_vision_language_datasets(dataset_configs, split, task):
    """
    Generate datasets based around caption, vqa and language.
    """

    datasets = []

    for dataset_config in tqdm(dataset_configs):
        name = dataset_config["name"]
        features = dataset_config.get("features", None)
        rename_columns = dataset_config.get("rename_columns", {})
        flatten = dataset_config.get("flatten", False)
        streaming = dataset_config.get("streaming", False)
        samples = dataset_config.get("samples", 10)

        dataset = load_dataset(
            name, split=split, streaming=streaming, trust_remote_code=True
        )

        if samples:
            dataset = dataset.shuffle(seed=42).take(samples)

        if flatten:  # YOU CAN'T DO THIS WITH ITERABLE DATASETS :(
            dataset = dataset.flatten()

        if features:
            dataset = dataset.select_columns(features)

        if rename_columns:
            dataset = dataset.rename_columns(rename_columns)

        if "image_url" in dataset.column_names:
            # fetch the images
            dataset = dataset.map(
                utils.fetch_images,
                batched=True,
                batch_size=100,
                fn_kwargs={"num_threads": 5},
            )  # change num_threads depending on the computer
            dataset = dataset.rename_column("image_url", "image")
        datasets.append(dataset)

    final_datasets = []
    for dataset in datasets:
        if isinstance(dataset, IterableDataset):
            dataset = Dataset.from_generator(
                partial(utils.gen_from_iterable_dataset, dataset),
                features=dataset.features,
            )
        final_datasets.append(dataset)
    dataset = concatenate_datasets(final_datasets)
    dataset.save_to_disk(f"MultiNet_{task}")
