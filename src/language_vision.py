"""
Generates the text and images dataset part of the MultiNet.
"""

from datasets import concatenate_datasets, load_dataset
from datasets import Value

LINKS = [
    # name, link, amount to sample
    # ("allenai/dolma", 100),
    ("OBELICS", "HuggingFaceM4/OBELICS", 100),
    ("coyo", "kakaobrain/coyo-700m", 100),
    ("MSCOCO", "shunk031/MSCOCO", 100),
    ("conceptual_captions", "conceptual_captions", 100),
    ("A-OKVQA", "HuggingFaceM4/A-OKVQA", 100),
    ("VQAv2", "HuggingFaceM4/VQAv2", 100),
]


def main():
    datasets = []
    for name, link, number_of_rows in LINKS:
        print(f"processing {name}")
        dataset = load_dataset(link, split='train',
                               streaming=True, trust_remote_code=True)
        dataset = dataset.shuffle(seed=42).take(number_of_rows)

        new_features = dataset.features.copy()
        if "question_id" in new_features:
            new_features["question_id"] = Value("string")

        dataset = dataset.cast(new_features)

        datasets.append(dataset)
    consolidated_dataset = concatenate_datasets(datasets)

    return consolidated_dataset


if __name__ == "__main__":
    main()
