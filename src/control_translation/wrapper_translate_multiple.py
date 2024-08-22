#Script to manage RAM utilization when translating and saving control data shards to disk 
import psutil
import subprocess
import time
import argparse
import sys

MAX_RAM_USAGE = 80  # Maximum RAM usage percentage
PYTHON_EXECUTABLE = "python"  # or "python3" depending on your system
MAIN_SCRIPT = "translatemultiple.py"

def get_ram_usage():
    return psutil.virtual_memory().percent

def run_main_program(dataset_name, dataset_path, output_dir):
    cmd = [
        PYTHON_EXECUTABLE, 
        MAIN_SCRIPT, 
        '--dataset_name', dataset_name,
        '--dataset_path', dataset_path,
        '--output_dir', output_dir
    ]
    process = subprocess.Popen(cmd)
    while True:
        ram_usage = get_ram_usage()
        if ram_usage > MAX_RAM_USAGE:
            print(f"RAM usage is {ram_usage}%. Restarting...")
            process.terminate()
            process.wait()
            return -1
        
        if process.poll() is not None:
            print("Process completed successfully.")
            return None  # Indicate that processing is complete
        
        time.sleep(10)  # Check RAM usage every 10 seconds

def main():
    parser = argparse.ArgumentParser(description='Wrapper to avoid RAM bottlenecks when translating control dataset shards.')
    parser.add_argument("--dataset_name", type=str, required=True, help="Mention the dataset in MultiNet that needs to be translated. Different datasets are: 'dm_lab_rlu', 'dm_control_suite_rlu', 'ale_atari', 'baby_ai', 'mujoco', 'vd4rl', 'meta_world', 'procgen', 'language_table', 'openx', 'locomuojoco' ")
    parser.add_argument("--dataset_path", type=str, required=True, help="Provide the path to the specified dataset file")
    parser.add_argument("--output_dir", type=str, required=True, help="Provide the path to store the translated dataset")
    parser.add_argument("--limit_schema", type=bool, default=False, help="Set to True if schema needs to be trimmed to [observations, actions, rewards]")
    parser.add_argument("--hf_test_data", type=bool, default=False, help="Set to True if test split from Huggingface JAT datasets needs to be returned along with train data")
    
    args = parser.parse_args()

    while True:
        start_index = run_main_program( args.dataset_name, args.dataset_path, args.output_dir)
        if start_index is None:
            break  # All shards processed
        time.sleep(60)  # Wait for 60 seconds before restarting to allow system cleanup

if __name__ == "__main__":
    main()