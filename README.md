<p align="center">
  <img src="assets/multinet_logo_square copy.png" alt="Multinet Logo" height="100" style="vertical-align: middle;">
  <h1 align="center" style="display: inline-block; vertical-align: middle; margin-left: 20px;">MultiNet: A Generalist Benchmark for Multimodal Action models</h1>
</p>

<p align="center">

  [![Website](https://img.shields.io/badge/Website-blue?style=flat-square&logo=home)](https://multinet.ai)
  [![Multinet v0.1 paper](https://img.shields.io/badge/Paper%20v0.1-green?style=flat-square&logo=read-the-docs)](https://multinet.ai/static/pdfs/Benchmarking%20Vision%20Language%20Action%20Models.pdf)
  [![Multinet v0.2 paper](https://img.shields.io/badge/Paper%20v0.2-arXiv-B31B1B?style=flat-square&logo=arXiv)](https://arxiv.org/abs/2505.05540)
  [![Dataset Spec paper](https://img.shields.io/badge/Dataset%20Spec-orange?style=flat-square&logo=database)](https://multinet.ai/static/pdfs/MultiNet_Dataset_Spec_Paper.pdf)
  [![GenESIS framework](https://img.shields.io/badge/GenESIS-blueviolet?style=flat-square&logo=github)](https://github.com/ManifoldRG/MultiNet/tree/main/src/modules)
  [![Contribute](https://img.shields.io/badge/Contribute%20via%20Discord-7289DA?style=flat-square&logo=discord)](https://discord.gg/Rk4gAq5aYr)

</p>

### This work is sponsored by, and is being done in close collaboration with [Metarch](https://metarch.ai/).

## 📢 Updates
2024-11-08: We release the first version of MultiNet where we profiled a SoTA VLM, SoTA VLA, and a novel Generalist model on OpenX Embodiment datasets - Multinet v0.1! Check our [website](https://multinet.ai) for more details.


## 🔍 Overview

This repo provides the following capabilities:
1. Download any or all datasets of what aims to be the largest consolidation of open-source vision-language + control/action (RL, Robotics) data
2. Translate control data of various formats and from various sources to a unified [Tensorflow Dataset format](https://www.tensorflow.org/datasets/api_docs/python/tfds). 
3. Evaluate the performance of GPT-4o, OpenVLA, and HuggingFace's JAT in a zero-shot setting on OpenX Embodiment datasets using the benchmark released in [Multinet v0.1](https://github.com/ManifoldRG/MultiNet/releases/tag/v0.1).
4. A [general framework](https://github.com/ManifoldRG/MultiNet/tree/main/src/modules) for mapping VLMs to other modality classes, with particular emphasis on action spaces. This framework allows one to adapt a wide range of models to multiple types of tasks or datasets for scaling effectively while reducing the amount of engineering effort required.  In MultiNet v0.1, GenESIS is used to evaluate GPT-4-o on the OpenX datasets.

As a part of Multinet v0.1, we also release [μGATO](https://github.com/eihli/mugato) - a small, simple, open-source implementation of what is described in DeepMind's GATO paper. This is our first step towards building a multimodal generalist action model.

<p align="center">
  <img src="assets/Multinet v0.1 release visual 3.0.png" alt="Multinet Figure" style="vertical-align: middle;">
</p>

## 🚀 Getting Started

#### To set up the environment for download, translation, and evaluation of GPT-4o and HuggingFace's JAT:

```bash
conda create -n multinet python=3.10
conda activate multinet
git clone https://github.com/ManifoldRG/MultiNet.git
cd MultiNet/src
pip install -r requirements.txt
```

#### To download the datasets in v0:

```bash
cd Multinet/src
python centralized_downloader --dataset_name <name of dataset you would like to download> --output_dir <directory where you would like to download the dataset>
```

#### To translate one file/shard of your desired control dataset (downloaded using the downloader script in this repo) to the TFDS format 

```bash
cd Multinet/src/control_translation
python centralized_translation --dataset_name <name of dataset whose file you would like to translate> --dataset_path <path to the downloaded dataset> --output_dir <directory where you would like to store the translated file>
```

#### To translate multiple files/shards of your desired control dataset (downloaded using the downloader script in this repo) to the TFDS format 

Make sure to modify the way the multiple files are being traversed for translation in translate_multiple.py in Multinet/src/control_translation according to your local file structure.

```bash
cd Multinet/src/control_translation
python wrapper_translate_multiple.py --dataset_name <name of dataset whose file you would like to translate> --dataset_path <path to the downloaded dataset> --output_dir <directory where you would like to store the translated files>
```

#### To evaluate JAT (in a zero-shot setting) on the 53 OpenX Embodiment datasets it was profiled on as a part of Multinet v0.1 

Make sure to set the path to the translated openx datasets you want to evaluate on and the path where you want to dump your results in Multinet/src/eval/profiling/jat/scripts/profile_openx.py

```bash
cd Multinet/src/eval/profiling/jat/scripts
python profile_openx.py
```

#### To evaluate GPT-4o (in a zero-shot setting) using the [GenESIS framework](https://github.com/ManifoldRG/MultiNet/tree/main/src/modules) on the 22 OpenX Embodiment datasets it was profiled on as a part of Multinet v0.1 

Make sure to adjust the path creation to the translated openx datasets you want to evaluate on, and the path where you want to dump your results in src/modules/dataset_modules/openx_module.py based on your local file structure 

```bash
cd Multinet/src/eval/eval_main.py
python eval_main.py --disk_root_dir <path to the translated openx datasets> --dataset_name openx --model gpt-4o-2024-05-13
```
Enter the OpenAI API key when prompted.

#### To evaluate OpenVLA (in a zero-shot setting) on the 20 OpenX Embodiment datasets it was profiled on as a part of Multinet v0.1 

We set up our conda environment and ran evaluations for OpenVLA on a GCP Instance with 1 L4 GPU, driver version 550.90.07, and CUDA version 12.4. For more details about the infrastructure used, refer to our [paper](https://multinet.ai/static/pdfs/Benchmarking%20Vision%20Language%20Action%20Models.pdf). If you are using our code out-of-the-box, we recommend using the same infrastructure.

For setup, create a new conda environment and run the OpenVLA environment setup bash script (this will download both the OpenVLA requirements as well as the broader MultiNet requirements):

```bash
cd Multinet/src
./openvla_multinet_setup.sh
```

To run evaluations:

```bash
cd Multinet
python src/eval/profiling/openvla/experiments/robot/openvla_profiling.py --profiling_dataset_folder_path <path to the translated datasets> --dataset_statistics_path src/eval/profiling/openvla/data/dataset_statistics.json --result_save_path <path to the directory where you would like to save your results>
```

## 📊 Evaluation and submission process to the Multinet benchmark

Here are steps to follow to evaluate your team's model on Multinet data and submit results to our benchmark:

*   Download the desired dataset using the download+translate SDK that we provide by following the steps mentioned above.
*   Open an issue on our Github with the tag `evaluate`. The issue title should be: “Add < your model name > to Multinet benchmark”. 
*  You can access the list of test episodes for a specific dataset at [src/eval/profiling/test_data](src/eval/profiling/test_data). These test episodes can then be translated from the downloaded data using the download+translate SDK by following the steps mentioned above.
*   We break down the required components to run evals using a model into 3 categories:
    *   **Ingestion pipeline**: Pipeline to feed the model with the test data with necessary input processing. This can be similar to the dataloaders we have implemented in [src/data_utils](src/data_utils)
    *   **Model adaptation**: Adapt your model to ingest the test data and produce actions in the appropriate format. This can be similar to how we have implemented model adaptations for various models such as [Genesis for VLMs](src/modules/), and custom adaptations for VLAs such as [OpenVLA](https://github.com/ManifoldRG/MultiNet/blob/main/src/eval/profiling/openvla/experiments/robot/openvla_profiling.py), [Pi0 Base](https://github.com/ManifoldRG/MultiNet/blob/main/src/eval/profiling/openpi/scripts/procgen_inference.py), and [Pi0 FAST](https://github.com/ManifoldRG/MultiNet/blob/main/src/eval/profiling/openpi/scripts/procgen_inference_fast.py)
    *   **Inference pipeline**: The inference pipeline should include a `predict_action` function that takes in the observations of a timestep/batch of timesteps as input, produces the action(s) for a given timestep/batch of timesteps, and processes it to ensure the outputs are in the appropriate format.
        *   **You must implement deterministic inference by setting explicit seed values for all stochastic operations within the model inference pipeline. This requirement applies to any component that introduces non-determinism during the inference process.**
*   You can then run inference on the test data to obtain zero-shot predictions for all the timesteps.
    *   Once all the predictions are obtained, they can be evaluated using the metrics we implement and report. You can find the helper functions that implement all the metrics in [src/eval_utils.py](src/eval_utils.py)
    *   The results should consist of a JSON file for each subdataset of the test dataset where the keys are the names of the metrics and the values are metric values. These JSON files should also contain the predictions of the model. You can use the results files in [src/v02_results/](src/v0_2results/) as reference.
*  Once the results are ready, you should open a PR that contains the code that can produce the results you report, and the final results in the correct format. Make sure to provide all necessary details for reproducibility - especially the seed values used in the inference pipeline. Associate this PR with the Issue opened in the second step mentioned above.
*   Upon review by our team, we will either merge the PR or request changes/modifications/clarifications.
    *   Once further queries are resolved, the PR will be merged and Issue closed.
*   Once the PR is merged and results are accepted, we will display the results on our [website](https://multinet.ai/)!


