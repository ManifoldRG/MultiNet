from src.modules.dataset_modules.procgen_module import ProcGenBatchModule
from src.modules.dataset_modules.openx_module import OpenXBatchModule

import argparse
import json

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="The directory of the configuration files.")
    parser.add_argument('--batch_job_info_path', type=str, required=True, help="The path to the batch job information.")
    parser.add_argument('--results_path', type=str, required=True, help="The path to save the results.")
    args = parser.parse_args()
    
    # Argument validation.
    # TODO: Update config.json everytime a new dataset or model is added into the new version.
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    assert args.dataset in config['datasets'].keys(), f"Specify the correct dataset name supported:\n{list(config['datasets'].keys())}"
    assert args.model in config["models"].keys(), f"Specify the correct model index supported.\n{list(config['models'].keys())}"

    # Setting the configurations of the current evaluation job.
    modality, source = config['models'][args.model]

    # TODO: More branches will be added during the implementation.
    dataset_module = None
    if args.dataset_family == 'procgen':
        dataset_module = ProcGenBatchModule(args.disk_root_dir, modality, source, args.model, 1, 0)
    elif args.dataset_family == 'openx':
        dataset_module = OpenXBatchModule(args.disk_root_dir, modality, source, args.model, 1, 0)
    
    with open(args.batch_job_info_path, 'r') as f:
        batch_info_dict = json.load(f)
    dataset_module.run_eval(args.results_path, batch_info_dict)