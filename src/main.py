"""
Call it with: python main.py --task=caption|language
"""

import json
import argparse

from tasks import get_datasets


def main(task):
    with open(task + ".json", "r") as f:
        file_contents = f.read()
        data = json.loads(file_contents)

    get_datasets(data["datasets"], data["split"], data["task"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MultiNet", description="Generates the datasets necessary for MultiNet. "
    )
    parser.add_argument(
        "--task",
        help="The task to be performed. It will generate a dataset for either language or caption",
    )
    args = parser.parse_args()
    main(args.task)
