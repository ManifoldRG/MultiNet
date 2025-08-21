# Script to manage RAM utilization when processing datasets to avoid memory bottlenecks
import psutil
import subprocess
import time
import argparse
import sys
from pathlib import Path

MAX_RAM_USAGE = 70  # Maximum RAM usage percentage
PYTHON_EXECUTABLE = "python"  # or "python3" depending on your system
MAIN_SCRIPT = "centralized_processor.py"

def get_ram_usage():
    return psutil.virtual_memory().percent

def run_main_program(input_dir, output_dir, list_datasets, process, process_all, ram_threshold=MAX_RAM_USAGE):
    cmd = [PYTHON_EXECUTABLE, MAIN_SCRIPT]
    
    # Add input and output directories
    if input_dir:
        cmd.extend(['--input-dir', str(input_dir)])
    if output_dir:
        cmd.extend(['--output-dir', str(output_dir)])
    
    # Add action arguments
    if list_datasets:
        cmd.append('--list')
    elif process:
        cmd.extend(['--process'] + process)
    elif process_all:
        cmd.append('--process-all')
    
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    
    while True:
        ram_usage = get_ram_usage()
        if ram_usage > ram_threshold:
            print(f"RAM usage is {ram_usage}%. Exceeds threshold of {ram_threshold}%. Restarting...")
            process.terminate()
            process.wait()
            return -1  # Indicate restart needed
        
        if process.poll() is not None:
            print("Processing completed successfully.")
            return None  # Indicate that processing is complete
        
        time.sleep(10)  # Check RAM usage every 10 seconds

def main():
    parser = argparse.ArgumentParser(description='Wrapper to avoid RAM bottlenecks when processing datasets.')
    parser.add_argument("--input-dir", type=Path, default=Path("./dataset_cache"),
                       help="Input directory (from downloader)")
    parser.add_argument("--output-dir", type=Path, default=Path("./processed_datasets"),
                       help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--process", nargs="+", help="Process specific datasets")
    parser.add_argument("--process-all", action="store_true", help="Process all available datasets")
    
    args = parser.parse_args()
    
    # Validate that at least one action is specified
    if not (args.list or args.process or args.process_all):
        print("Error: Must specify one of --list, --process, or --process-all")
        parser.print_help()
        sys.exit(1)
    
    # If --list is specified, run once without restart logic since it's quick
    if args.list:
        # For listing, we allow higher RAM usage since it's a quick read-only operation
        return_code = run_main_program(args.input_dir, args.output_dir, args.list, args.process, args.process_all, ram_threshold=90)
        
        if return_code is None:
            print("Dataset listing completed.")
        else:
            print("Dataset listing failed.")
        return
    
    # For processing operations, use restart logic
    restart_count = 0
    max_restarts = 10  # Prevent infinite restart loops
    
    while restart_count < max_restarts:
        return_code = run_main_program(args.input_dir, args.output_dir, args.list, args.process, args.process_all)
        
        if return_code is None:
            print("Dataset processing completed successfully.")
            break  # Processing completed successfully
        else:
            restart_count += 1
            print(f"Restarting process (attempt {restart_count}/{max_restarts})...")
            time.sleep(60)  # Wait for 60 seconds before restarting to allow system cleanup
    
    if restart_count >= max_restarts:
        print(f"Maximum restart attempts ({max_restarts}) reached. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main() 