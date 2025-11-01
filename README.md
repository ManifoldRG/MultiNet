<p align="center">
  <kbd>
  <img src="assets/multinet_logo_square copy.png" alt="Multinet Logo" style="height:200px; border-radius:50%;">
  <h1 align="center" style="display: inline-block; vertical-align: middle; margin-left: 20px;">MultiNet: A Generalist Benchmark for the Next Generation of Multimodal Models</h1>
  </kbd>
</p>

<p align="center">
  <a href="https://multinet.ai/"><img src="https://img.shields.io/badge/Website-blue?style=flat-square&logo=googlechrome" alt="Website"></a> 
  <a href="https://multinet.ai/"><img src="https://img.shields.io/badge/Multinet%20v1.0-Release-blue?style=flat-square&logo=Blogger" alt="Multinet v1.0 release"></a>
  <a href="https://arxiv.org/abs/2505.05540"><img src="https://img.shields.io/badge/Multinet%20v0.2%20paper-arXiv-B31B1B?style=flat-square&logo=arXiv" alt="Multinet v0.2 paper"></a> 
  <a href="https://arxiv.org/abs/2411.05821"><img src="https://img.shields.io/badge/Multinet%20v0.1%20paper-arXiv-B31B1B?style=flat-square&logo=arXiv" alt="Multinet v0.1 paper"></a> 
  <a href="https://github.com/ManifoldRG/MultiNet/tree/main/src/modules"><img src="https://img.shields.io/badge/GenESIS%E2%A0%80%E2%A0%80%E2%A0%80%E2%A0%80%E2%A0%80%E2%A0%80%E2%A0%80-blueviolet?style=flat-square&logo=github" alt="GenESIS framework"></a> 
  <a href="https://discord.gg/Rk4gAq5aYr"><img src="https://img.shields.io/badge/Contribute%E2%A0%80%E2%A0%80%E2%A0%80%E2%A0%80%E2%A0%80-7289DA?style=flat-square&logo=discord" alt="Contribute"></a>
</p>

### MultiNet is a collaborative initiative with contributions from leading research teams at institutions like:

<p align="center">
  <a href="https://metarch.ai/" target="_blank">
    <kbd>
    <img src="assets/metarchlogo.jpg" alt="Metarch.ai Logo" height="40">
    </kbd>
  </a>
  <a href="https://www.manifoldrg.com/" target="_blank">
    <kbd>
    <img src="assets/manifold_logo_square.png" alt="Manifold Research Logo" height="40">
    </kbd>
  </a>
  <a href="https://www.mit.edu/" target="_blank">
    <kbd>
    <img src="assets/mitlogo.jpg" alt="MIT Logo" height="40">
    </kbd>
  </a>
  <a href="https://www.gatech.edu/" target="_blank">
    <kbd>
    <img src="assets/gtlogo_alt.png" alt="Georgia Tech Logo" style="height:40px; border-radius:50%;">
    </kbd>
  </a>
  <a href="https://www.tufts.edu/" target="_blank">
    <kbd>
    <img src="assets/tuftslogo.jpg" alt="Tufts Logo" style="height:40px; border-radius:50%;">
    </kbd>
  </a>
</p>

### _Need to Run Evaluations on Production Multimodal, Computer Use, or Robotics AI System? [We can help!](https://forms.gle/DuMyjoZrEYR641ro6)_

## 📢 Updates
- 🌟 2025-13-10: Multinet v1.0 - We release our most comprehensive benchmark yet - evaluating a SoTA VLM, VLA, and generalist model on a wide variety of multimodal understanding and action datasets. Read more [here](https://multinet.ai/static/pages/Multinetv1.html)
- 🏅 2025-06-10: Paper accepted at ICML 2025! Our paper detailing the Open-Source contributions of Multinet that benefit the AI community has been accepted at the [CodeML Workshop](https://codeml-workshop.github.io/codeml2025/) at ICML 2025! Read our paper [here](https://multinet.ai/static/pdfs/An%20Open-Source%20Software%20Toolkit%20&%20Benchmark%20Suite%20for%20the%20Evaluation%20and%20Adaptation%20of%20Multimodal%20Action%20Models.pdf).
- 🏆 2025-05-22: Multinet v0.2 - We systematically profile state-of-the-art VLAs and VLMs to understand how they perform in procedurally generated OOD game environments! Read more about our release [here](https://multinet.ai/static/pages/Multinetv02.html)
- 🎉 2024-11-08: We release the first version of MultiNet where we profiled SoTA VLMs and VLAs on real-world robotics tasks - Multinet v0.1! Check our [release page](https://multinet.ai/static/pages/Multinetv01.html) for more details.
- 🚀 2024-03-22: Introducing Multinet! A new generalist benchmark to evaluate Vision-Language & Action models. Learn more [here](https://multinet.ai)

## 🔍 Overview

This repo provides the following:
1. Ability to profile VLMs and VLAs on our generalist evaluation framework with a comprehensive coverage of open-source vision-language + control/action (RL, Robotics) tasks
2. Ability to translate control data of various formats and from various sources to a unified [Tensorflow Dataset format](https://www.tensorflow.org/datasets/api_docs/python/tfds). 
3. Evaluate the performance of SoTA VLMs and VLAs such as GPT-5, Pi0, and Magma in a zero-shot setting on a wide-variety of tasks detaied [here](https://multinet.ai/static/pages/Multinetv1.html).
4. A [general framework](https://github.com/ManifoldRG/MultiNet/tree/main/src/v1/modules) for mapping VLMs to other modality classes, with particular emphasis on action spaces. This framework allows one to adapt a wide range of models to multiple types of tasks or datasets for scaling effectively while reducing the amount of engineering effort required.  In MultiNet v1.0, GenESIS is used to evaluate GPT 5 on the OpenX, Overcooked, PIQA, ODINW, and SQA3D datasets.
5. Test splits of Multinet datasets and clear guidelines required to evaluate your model on them and add the results to our leaderboard in order to compare performance against SoTA VLMs and VLAs.

Also related to the MultiNet effort is <a href="https://github.com/eihli/mugato"><img src="https://img.shields.io/badge/%CE%BCGATO%E2%A0%80%E2%A0%80-dimgray?style=flat-square&logo=github" alt="μGATO on GitHub" style="vertical-align: middle;"></a> - a small, simple, open-source implementation of what is described in DeepMind's GATO paper. This project marks our initial step towards building a multimodal generalist action model.

<br>

<p align="center">
  <img src="assets/Multinet v1 release visual final.png" alt="Multinet v1.0 Figure" style="width: 700px; height: auto; vertical-align: middle;">
</p>

<!-- <br>

<p align="center">
  <img src="assets/v0_2_visual.jpg" alt="Multinet v0.2 Figure" style="width: 700px; height: auto; vertical-align: middle;">
</p>

<br>

<p align="center">
  <img src="assets/Multinet v0.1 release visual 3.0.png" alt="Multinet Figure" style="width: 700px; height: auto; vertical-align: middle;">
</p> -->

## 🚀 Getting Started

#### To set up the environment for Multinet:

```bash
conda create -n multinet python=3.10
conda activate multinet
git clone https://github.com/ManifoldRG/MultiNet.git
cd MultiNet/src
pip install -r requirements.txt
```

#### To download the datasets in v1:

```bash
cd Multinet/src/v1
python centralized_downloader --download <name of dataset you would like to download>
```

#### To translate one file/shard of your desired control dataset (downloaded using the downloader script in this repo) to the TFDS format 

```bash
cd Multinet/src/v1
python centralized_processor --input_dir <path to the downloaded dataset> --output_dir <directory where you would like to store the translated file>
```

#### To translate multiple files/shards of your desired control dataset (downloaded using the downloader script in this repo) to the TFDS format 

> **Note:** Make sure to modify the way the multiple files are being traversed for translation in translate_multiple.py in Multinet/src/control_translation according to your local file structure.

```bash
cd Multinet/src/v1
python wrapper_centralized_processor.py --input_dir <path to the downloaded dataset> --output_dir <directory where you would like to store the translated file>
```

#### To evaluate models on MultiNet datasets

We provide comprehensive evaluation guides for different models:

**Magma Model Evaluation:**
For detailed instructions on evaluating Magma on ODINW, PIQA, SQA3D, RoboVQA, Overcooked, BFCL, and OpenX datasets, see the [Magma Evaluation Guide](docs/magma_evaluation.md).

**Pi0 Base Model Evaluation:**
For detailed instructions on evaluating Pi0 Base on ODINW, PIQA, SQA3D, RoboVQA, BFCL, Overcooked, and OpenX datasets, see the [Pi0 Evaluation Guide](docs/pi0_evaluation.md).

**GPT Model Evaluation (GenESIS Framework):**
For detailed instructions on evaluating GPT-5 using the GenESIS framework on ODINW, PIQA, SQA3D, RoboVQA, Overcooked, and OpenX datasets, see the [GenESIS Evaluation Guide](docs/genesis_evaluation.md).

## 📊 Process for Submission to the MultiNet Benchmark

### Using the Evaluation Harness

We provide a standardized evaluation harness for benchmarking your model on MultiNet datasets. The harness provides:

- **Standardized Interface**: Create model adapters that inherit from the base `ModelAdapter` class
- **Dockerized Evaluation**: Reproducible evaluations in isolated containers
- **Various Task Types**: Support for datasets that span VQA, action prediction, function calling, and more

**Quick Start:**
1. Create your model adapter(s) by inheriting from `ModelAdapter` in `src/eval_harness/model_adapter.py`
2. Test your model adapter using the `scripts/eval_harness/evaluate.py` entrypoint which loads sample data in a standard format
2. Configure `harness_dataset_config.txt` and `Dockerfile` with your adapter settings
3. Run `./build_and_run_eval_container.sh DATASET_NAME`

**For complete instructions, see the [Evaluation Harness Guide](src/eval_harness/README.md).**

For questions or assistance, contact us directly or join our [Discord community](https://discord.gg/Rk4gAq5aYr).

## 📜 Citation

If you use MultiNet in your research, please cite our work:

```bibtex

Multinet v0.2 - Benchmarking Vision, Language, & Action Models in Procedurally Generated, Open Ended Action Environments

@misc{guruprasad2025benchmarkingvisionlanguage,
      title={Benchmarking Vision, Language, & Action Models in Procedurally Generated, Open Ended Action Environments}, 
      author={Pranav Guruprasad and Yangyue Wang and Sudipta Chowdhury and Harshvardhan Sikka},
      year={2025},
      eprint={2505.05540},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.05540}, 
      }

Multinet v0.1 - Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks

@misc{guruprasad2024benchmarkingvisionlanguage,
      title={Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks}, 
      author={Pranav Guruprasad and Harshvardhan Sikka and Jaewoo Song and Yangyue Wang and Paul Pu Liang},
      year={2024},
      eprint={2411.05821},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.05821},
      }

Multinet Vision and Dataset specification

@misc{guruprasad2024benchmarking,
      author={Guruprasad, Pranav and Sikka, Harshvardhan and Song, Jaewoo and Wang, Yangyue and Liang, Paul},
      title={Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks},
      DOI={10.20944/preprints202411.0494.v1},
      year={2024},
      }    

```


