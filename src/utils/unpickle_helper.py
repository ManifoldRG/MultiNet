#!/usr/bin/env python
"""
Helper script to unpickle a file with a specific pandas version and save it as CSV.

This script is intended to be run in a separate virtual environment with a
specific pandas version to handle version incompatibilities in pickle files.
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Unpickle a pandas DataFrame and save it as CSV.")
    parser.add_argument("input_path", help="Path to the input pickle file.")
    parser.add_argument("output_path", help="Path to save the output CSV file.")
    args = parser.parse_args()

    # Load the DataFrame from the old pickle file
    data = pd.read_pickle(args.input_path)

    # Save the DataFrame to a CSV file
    data.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    main()
