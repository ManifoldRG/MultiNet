import datasets
from datasets import load_dataset
import os
from argparse import ArgumentParser

def build_arg_parser() -> ArgumentParser:

    parser = ArgumentParser(description=f'Download the dataset')
    parser.add_argument("--dataset_name", type=str, required=True, help="Mention the vision language datasets that are part of the MultiNet eval collection. Different datasets are: 'ms_coco_captions', 'conceptual_captions', 'flickr', 'a_okvqa','vqa_v2', 'text_vqa', 'vizwiz', 'winogavil', 'imagenet_r', 'objectnet', 'hellaswag', 'winogrande', 'arc', 'commonsenseqa', 'mmlu'")
    parser.add_argument("--output_dir", type=str, required=True, help="Provide the path to store the downloaded dataset")
    return parser


def dl_eval(dataset_name: str, output_dir: str):

    #Download the specified dataset from HuggingFace
    if dataset_name == 'ms_coco_captions':
        ds = load_dataset("HuggingFaceM4/COCO", trust_remote_code=True)
    elif dataset_name == 'conceptual_captions':
        ds = load_dataset("google-research-datasets/conceptual_captions", trust_remote_code=True)
    elif dataset_name == 'flickr':
        ds = load_dataset("nlphuji/flickr30k", trust_remote_code=True) #Only the test set
    elif dataset_name == 'a_okvqa':
        ds = load_dataset("HuggingFaceM4/A-OKVQA", trust_remote_code=True)
    elif dataset_name == 'vqa_v2':
        ds = load_dataset('HuggingFaceM4/VQAv2', trust_remote_code=True)
    elif dataset_name == 'text_vqa':
        ds = load_dataset('textvqa', trust_remote_code=True)
    elif dataset_name == 'vizwiz':
        ds = load_dataset("lmms-lab/VizWiz-VQA", trust_remote_code=True)
    elif dataset_name == 'winogavil':
        ds = load_dataset("nlphuji/winogavil", trust_remote_code=True, streaming=True)
    elif dataset_name == 'imagenet_r':
        ds = load_dataset("axiong/imagenet-r", trust_remote_code=True)
    elif dataset_name == 'objectnet':
        ds = load_dataset("timm/objectnet", trust_remote_code=True, token= os.environ['HF_AUTH_TOKEN']) #Set HF auth token
    elif dataset_name == 'hellaswag':
        ds = load_dataset("Rowan/hellaswag", trust_remote_code=True)
    elif dataset_name == 'winogrande':
        ds = load_dataset("allenai/winogrande",  'winogrande_s', trust_remote_code=True)
    elif dataset_name == 'arc':
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", trust_remote_code=True)
    elif dataset_name == 'commonsenseqa':
        ds = load_dataset("tau/commonsense_qa", trust_remote_code=True)
    elif dataset_name == 'mmlu':
        ds = load_dataset("cais/mmlu", "all", trust_remote_code=True)
    

    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
    ds.save_to_disk(os.path.join(output_dir, dataset_name))
    print('Successfully downloaded and saved')
    return

if __name__ == "__main__":
    
    parser = build_arg_parser()
    args = parser.parse_args()
    dl_eval(args.dataset_name, args.output_dir) 
