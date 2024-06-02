"""
Call it with: python main.py --task=caption|language|vqa
"""

import os
import json
import argparse

from tasks import get_datasets


def main(task):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"{task}.json")

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            file_contents = f.read()
            data = json.loads(file_contents)
            get_datasets(data["datasets"], data["split"], data["task"])
    else:
        print(f"File {file_path} not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MultiNet", description="Generates the datasets necessary for MultiNet. "
    )
    parser.add_argument(
        "--task",
        help="The task to be performed. It will generate a dataset for either language, caption or vqa.",
        required=True,
    )
    args = parser.parse_args()
    main(args.task)
