import os, sys

# Adding the root directory to the system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

from src.modules.dataset_modules.procgen_module import ProcGenBatchModule
from src.modules.dataset_modules.openx_module import OpenXBatchModule
from src.v1.modules.overcooked_module import OvercookedBatchModule
from src.v1.modules.odinw_module import ODinWBatchModule
from src.v1.modules.sqa3d_module import SQA3DBatchModule
from src.v1.modules.robovqa_module import RoboVQABatchModule

import datetime
import argparse
import json


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, required=True, help="The root directory of the translated data.")
    parser.add_argument('--dataset_family', type=str, required=True, help="The name of the dataset to evaluate.")
    parser.add_argument('--dataset_name', type=str, required=True, help="The name of the dataset to evaluate.")
    parser.add_argument('--model', type=str, required=True, help="The name of the model to evaluate.")
    parser.add_argument("--metadata_dir", type=str, required=True, help="The directory to save batch info in after sending the jobs.")
    parser.add_argument('--batch_size', type=int, default=1, help="The batch size used for evaluation.")
    parser.add_argument('--k_shots', type=int, default=0, help="Setting how many few-shots examples should be used.")

    args = parser.parse_args()

    
    # Argument validation.
    # TODO: Update config.json everytime a new dataset or model is added into the new version.
    config_path = os.path.join(ROOT_DIR, 'src', 'config.json')
    assert os.path.exists(config_path), f"The config file does not exist: {config_path}"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    assert args.dataset_family in config['datasets'].keys(), f"Specify the correct dataset name supported:\n{list(config['datasets'].keys())}"
    assert args.model in config["models"].keys(), f"Specify the correct model index supported.\n{list(config['models'].keys())}"
    
    # Setting the configurations of the current evaluation job.
    modality, source = config['models'][args.model]

    confirm_intention = input(f"This script will process all timesteps for {args.dataset_family} {args.dataset_name} in {args.data_root_dir} and send batch jobs to the {source} API. Do you want to continue? (y/n): ")
    if confirm_intention.lower() != 'y':
        print("Exiting the script.")
        exit(0)
        
    # Setting the extra information depending on the model.
    if source == 'openai':
        os.environ["OPENAI_API_KEY"] = input("Enter the OpenAI API key: ")

    data_root_dir = os.path.abspath(args.data_root_dir)
    assert os.path.exists(data_root_dir), f"The data root directory does not exist: {data_root_dir}"
    assert os.path.isdir(data_root_dir), f"The data root directory is not a directory: {data_root_dir}"
    
    dataset_module = None
    if args.dataset_family == 'procgen':
        dataset_module = ProcGenBatchModule(data_root_dir, modality, source, args.model, os.path.abspath(args.metadata_dir), args.batch_size, args.k_shots)
    elif args.dataset_family == 'openx':
        dataset_module = OpenXBatchModule(data_root_dir, modality, source, args.model, os.path.abspath(args.metadata_dir), args.batch_size, args.k_shots)
    elif args.dataset_family == 'overcooked_ai':
        dataset_module = OvercookedBatchModule(data_root_dir, modality, source, args.model, os.path.abspath(args.metadata_dir), args.batch_size, args.k_shots)
    elif args.dataset_family == 'odinw':
        dataset_module = ODinWBatchModule(data_root_dir, modality, source, args.model, os.path.abspath(args.metadata_dir), args.batch_size, args.k_shots)
    elif args.dataset_family == 'sqa3d':
        dataset_module = SQA3DBatchModule(data_root_dir, modality, source, args.model, os.path.abspath(args.metadata_dir), args.batch_size, args.k_shots)
    elif args.dataset_family == 'robot_vqa':
        dataset_module = RoboVQABatchModule(data_root_dir, modality, source, args.model, os.path.abspath(args.metadata_dir), args.batch_size, args.k_shots)
    else:
        print(f"The dataset family {args.dataset_family} is not supported.")
        exit(1)
        
    assert dataset_module is not None, "The dataset module has not been set correctly. Check required."
    batch_list = dataset_module._send_batch_jobs_for_dataset(args.dataset_name)
    batch_list = {args.dataset_name: batch_list}
    with open(f"{args.dataset_family}_{args.dataset_name}_batch_list_{datetime.datetime.now()}.json", 'w') as f:
        json.dump(batch_list, f)

    
