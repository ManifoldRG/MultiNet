"""
Call it with: python main.py --task=caption|language|vqa|control
"""

import os
import json
import argparse

from tasks import get_control_datasets, get_vision_language_datasets


def main(task):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, f"data_config/{task}.json")

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            file_contents = f.read()
            data = json.loads(file_contents)
            if data["task"] == "control":
                get_control_datasets(data["datasets"], data["split"])
            else:
                get_vision_language_datasets(data["datasets"], data["split"], data["task"])
    else:
        print(f"File {file_path} not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MultiNet", description="Generates the datasets necessary for MultiNet. "
    )
    parser.add_argument(
        "--task",
        help="The task to be performed. It will generate a dataset for either language, caption, vqa or control.",
        required=True,
    )
    args = parser.parse_args()
    main(args.task)
