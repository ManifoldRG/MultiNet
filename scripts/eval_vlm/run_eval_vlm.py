import os, sys

# Adding the root directory to the system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

from src.modules.dataset_modules.procgen_module import ProcGenBatchModule
from src.modules.dataset_modules.openx_module import OpenXBatchModule
from src.v1.modules.overcooked_module import OvercookedBatchModule, OvercookedModule
from src.modules.dataset_modules.openx_module import OpenXBatchModule, OpenXModule
from src.v1.modules.robovqa_module import RoboVQABatchModule, RoboVQAModule
from src.v1.modules.piqa_module import PIQABatchModule, PIQAModule
from src.v1.modules.odinw_module import ODinWBatchModule


import argparse
import json
import numpy as np 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_process', type=bool, default=False, help="Whether to run inference using the batch processing api")
    parser.add_argument('--batch_job_info_path', type=str, help="The path to the batch job information.")
    parser.add_argument('--results_path', required=True, type=str, help="A JSON file path to save the results.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to evaluate.")
    parser.add_argument('--batch_size', type=int, default=1, help="The batch size used for evaluation.")
    parser.add_argument('--k_shots', type=int, default=0, help="Setting how many few-shots examples should be used.")
    parser.add_argument('--dataset_name', type=str, required=True, help="The name of the dataset to evaluate.")
    parser.add_argument('--dataset_family', type=str, help="The name of the dataset family to evaluate.")
    parser.add_argument('--disk_root_dir', type=str, required=True, help="The root directory of the translated data.")
    args = parser.parse_args()

    # Argument validation.
    # TODO: Update config.json everytime a new dataset or model is added into the new version.
    config_path = os.path.join(ROOT_DIR, 'src', 'config.json')
    assert os.path.exists(config_path), f"The config file does not exist: {config_path}"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if args.batch_process:
        assert args.batch_job_info_path is not None, "The batch job information path is required when running batch processing."
        with open(args.batch_job_info_path, 'r') as f:
            batch_info_dict = json.load(f)
    
        a_fp = next(iter(batch_info_dict.values()))[0]
        assert os.path.exists(a_fp), f"The batch job information file does not exist: {a_fp}"
        
        batch_info = np.load(a_fp, allow_pickle=True)

        dataset_family = batch_info['dataset_family'].item()
        model = batch_info['model'].item()

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
            dataset_module = ProcGenBatchModule(args.disk_root_dir, modality, source, model, 1, 0)
        elif dataset_family == 'openx' and args.batch_process:
            dataset_module = OpenXBatchModule(args.disk_root_dir, modality, source, model, 1, 0)
        elif dataset_family == 'overcooked_ai':
            dataset_module = OvercookedBatchModule(args.disk_root_dir, modality, source, model, 1, 0)
        elif dataset_family == "robot_vqa" and args.batch_process:
            dataset_module = RoboVQABatchModule(args.disk_root_dir, modality, source, model, 1, 0)
        elif dataset_family == "piqa" and args.batch_process:
            dataset_module = PIQABatchModule(args.disk_root_dir, modality, source, model, 1, 0)
        elif dataset_family == "odinw" and args.batch_process:
            dataset_module = ODinWBatchModule(args.disk_root_dir, modality, source, model, args.dataset_name, args.batch_job_info_path,  1, 0)
        dataset_module.run_eval(os.path.abspath(args.results_path), batch_info_dict)
        
    

   
    assert args.dataset_family is not None, "The dataset family is required when running single input inference."
    
    if not args.batch_process:
        os.environ["OPENAI_API_KEY"] = input("Enter the OpenAI API key: ")
        
        if args.dataset_family == 'openx':
            dataset_module = OpenXModule(args.disk_root_dir, 'vlm', 'openai', args.model, args.dataset_name, 1, 0)
        elif args.dataset_family == 'overcooked_ai':
            os.environ["OPENAI_API_KEY"] = input("Enter the OpenAI API key: ")
            dataset_module = OvercookedModule(args.disk_root_dir, 'vlm', 'openai', args.model, args.dataset_name, 1, 0)
        elif args.dataset_family == 'robot_vqa':
            os.environ["OPENAI_API_KEY"] = input("Enter the OpenAI API key: ")
            dataset_module = RoboVQAModule(args.disk_root_dir, 'vlm', 'openai', args.model, args.dataset_name, 1, 0)
        elif args.dataset_family == 'piqa':
            os.environ["OPENAI_API_KEY"] = input("Enter the OpenAI API key: ")
            dataset_module = PIQAModule(args.disk_root_dir, 'vlm', 'openai', args.model, args.dataset_name, 1, 0)
        
        dataset_module.run_eval(args.results_path)
    
    