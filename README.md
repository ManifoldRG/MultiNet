<p align="center">
  <img src="@Multinetlogo.png" alt="Multinet Logo" height="100" style="vertical-align: middle;">
  <h1 align="center" style="display: inline-block; vertical-align: middle; margin-left: 20px;">MultiNet: A Generalist Benchmark for Vision-Language & Action models</h1>
</p>

<p align="center">
  <a href="https://multinet.ai">Website</a> •
  <a href="https://arxiv.org/abs/">Paper</a> •
  <a href="https://arxiv.org/abs/">Dataset Spec</a> •
  <a href="https://github.com/ManifoldRG/MultiNet/tree/main/src/modules">GenESIS framework</a> •
  <a href="https://discord.gg/D5YnaQm7">Discord</a>
</p>


### This work is sponsored by, and is being done in close collaboration with [Metarch](https://metarch.ai/).

## 📢 Updates
2024-11-05: We release the first version of MultiNet where we profiled a SoTA VLM, SoTA VLA, and SoTA Generalist model on OpenX Embodiment datasets - Multinet v0.1! Check our [website](https://multinet.ai) for more details.

## 🔍 Overview

This repo provides the following capabilities:
1. Download any or all datasets of what aims to be the largest consolidation of open-source vision-language + control/action (RL, Robotics) data
2. Translate control data of various formats and from varioussources to a unified [Tensorflow Dataset format](https://www.tensorflow.org/datasets/api_docs/python/tfds). 
3. Evaluate the performance of GPT-4o, OpenVLA, and HuggingFace's JAT in a zero-shot setting on 20 OpenX Embodiment datasets using the benchmark released in [Multinet v0.1](https://github.com/ManifoldRG/MultiNet/releases/tag/v0.1).
4. A [general framework](https://github.com/ManifoldRG/MultiNet/tree/main/src/modules) for mapping VLMs to other modality classes, with particular emphasis on action spaces. This framework allows one to adapt a wide range of models to multiple types of tasks or datasets for scaling effectively while reducing the amount of engineering effort required.  In MultiNet v0.1, GenESIS is used to evaluate GPT-4-o on the OpenX datasets.

We also provide [μGATO](https://github.com/eihli/mugato) - a small, simple, open-source implementation of what is described in DeepMind's GATO paper. This is our first step towards building a multimodal generalist action model.

## 🚀 Getting Started

To set up the environment for download, translation, and evaluation of GPT-4o and HuggingFace's JAT

```bash
conda create -n multinet python=3.10
conda activate multinet
git clone https://github.com/ManifoldRG/MultiNet.git
cd MultiNet/src
pip install -r requirements.txt
```

To download the datasets in v0

```bash
cd Multinet/src
python centralized_downloader --dataset_name <name of dataset you would like to download> --output_dir <directory where you would like to download the dataset>
```

To translate one file/shard of your desired control dataset (downloaded using the downloader script in this repo) to the TFDS format 

```bash
cd Multinet/src/control_translation
python centralized_translation --dataset_name <name of dataset whose file you would like to translate> --dataset_path <path to the downloaded dataset> --output_dir <directory where you would like to store the translated file>
```

To translate multiple files/shards of your desired control dataset (downloaded using the downloader script in this repo) to the TFDS format 

```bash
cd Multinet/src/control_translation
python wrapper_translate_multiple.py --dataset_name <name of dataset whose file you would like to translate> --dataset_path <path to the downloaded dataset> --output_dir <directory where you would like to store the translated file>
```




