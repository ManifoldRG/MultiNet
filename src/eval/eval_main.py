import os
import sys
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

from src.modules.dataset_modules.openx_module import OpenXModule

import argparse
import json


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="The name of the dataset to evaluate.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to evaluate.")

    args = parser.parse_args()

    # Argument validation.
    # TODO: Update config.json everytime a new dataset or model is added into the new version.
    with open(f"{ROOT_DIR}/src/config.json", 'r') as f:
        config = json.load(f)
    assert args.dataset in config['datasets'].keys(), f"Specify the correct dataset name supported:\n{list(config['datasets'].keys())}"
    assert args.model in config["models"].keys(), f"Specify the correct model index supported.\n{list(config['models'].keys())}"

    # Setting the configurations of the current evaluation job.
    modality, source = config['models'][args.model]

    # TODO: More branches will be added during the implementation.
    dataset_module = None
    if args.dataset == 'openx':
        dataset_module = OpenXModule(modality, source)

    assert dataset_module is not None, "The dataset module has not been set correctly. Check required."
