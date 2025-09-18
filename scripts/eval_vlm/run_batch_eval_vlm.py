import os, sys

# Adding the root directory to the system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

from src.modules.dataset_modules.procgen_module import ProcGenBatchModule
from src.modules.dataset_modules.openx_module import OpenXBatchModule
from src.v1.modules.overcooked_module import OvercookedBatchModule
from src.v1.modules.robovqa_module import RoboVQABatchModule
from src.v1.modules.piqa_module import PIQABatchModule
from src.v1.modules.odinw_module import ODinWBatchModule

import argparse
import json
import numpy as np 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_job_info_path', type=str, required=True, help="The json file path to the batch list, typically with a timestamp in the file name.")
    parser.add_argument('--results_path', type=str, required=True, help="A JSON file path to save the results.")
    args = parser.parse_args()
    
    with open(args.batch_job_info_path, 'r') as f:
        batch_info_dict = json.load(f)
    
    a_fp = next(iter(batch_info_dict.values()))[0]
    assert os.path.exists(a_fp), f"The batch job information file does not exist: {a_fp}"
    
    batch_info = np.load(a_fp, allow_pickle=True)

    dataset_family = batch_info['dataset_family'].item()
    model = batch_info['model'].item()
    
    # Argument validation.
    # TODO: Update config.json everytime a new dataset or model is added into the new version.
    config_path = os.path.join(ROOT_DIR, 'src', 'config.json')
    assert os.path.exists(config_path), f"The config file does not exist: {config_path}"
    with open(config_path, 'r') as f:
        config = json.load(f)

    assert dataset_family in config['datasets'].keys(), f"Specify the correct dataset name supported:\n{list(config['datasets'].keys())}"
    assert model in config["models"].keys(), f"Specify the correct model index supported.\n{list(config['models'].keys())}"

    # Setting the configurations of the current evaluation job.
    modality, source = config['models'][model]

    # Setting the extra information depending on the model.
    if source == 'openai':
        os.environ["OPENAI_API_KEY"] = input("Enter the OpenAI API key: ")
        
    # TODO: More branches will be added during the implementation.
    dataset_module = None
    if dataset_family == 'procgen' and args.batch_process:
        dataset_module = ProcGenBatchModule('', modality, source, model, '')
    elif dataset_family == 'openx' and args.batch_process:
        dataset_module = OpenXBatchModule('', modality, source, model, '')
    elif dataset_family == 'overcooked_ai':
        dataset_module = OvercookedBatchModule('', modality, source, model, '')
    elif dataset_family == "robot_vqa" and args.batch_process:
        dataset_module = RoboVQABatchModule('', modality, source, model, '')
    elif dataset_family == "piqa" and args.batch_process:
        dataset_module = PIQABatchModule('', modality, source, model, '')
    elif dataset_family == "odinw" and args.batch_process:
        dataset_module = ODinWBatchModule('', modality, source, model, '')
    dataset_module.run_eval(os.path.abspath(args.results_path), batch_info_dict)