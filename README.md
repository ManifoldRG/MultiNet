# MultiNet

We are currently working towards a Multi Modal Generalist Benchmark. The first
version of the project is going to be a pretraining corpus for the
[NEKO](https://github.com/ManifoldRG/Neko) project.

The main idea behind this pretraining corpus is the original NEKO dataset, but
we do not have access to all of the data that Google DeepMind had because most
of it was close sourced. That is why we are using open source alternatives.

The main pretraining corpus of the original GATO project is the following:

![The original gato corpus](./assets/gato_corpus.png)

This is our complimentary view of what each dataset would be replaced with for the Vision / Language datasets.

| Vision/language dataset | Weight | Replaced by                            |
| ----------------------- | ------ | ---------------------------------------|
| MassiveText             | 6.70%  | FineWeb Edu                            |
| M3W                     | 4.00%  | OBELICS                                |
| ALIGN                   | 0.67%  | COYO-700M                              |
| MS-COCO                 | 0.67%  | Open Source                            |
| Conceptual Captions     | 0.67%  | Open Source                            |
| LTIP                    | 0.67%  | Datacomp-1B                            |
| OKVQA                   | 0.67%  | AOKVQA (The augmented version of OKVQA)|
| VQA-V2                  | 0.67%  | Open Source                            |
| Total Vision + Language | 14.7%  |                                        |


As for the Control datasets, the following list is what we are using:



Finally, we are planning to evaluate part of the following list:

| Model                       | Modalities                                                |
|-----------------------------|-----------------------------------------------------------|
| Uni-Perceiver               | Text, Images, Video                                       |
| Unified-10                  | Text, Images, Video                                       |
| OFA+                        | Text, Images, Video, Audio                                |
| mPLUG-2                     | Text, Images, Video, Audio                                |
| Meta-Transformer            | Text, Infrared, Hyper-spectrum, IMU, Graph, Time series   |
| NEXT-GPT                    | Text, Images, Video, Audio                                |
| OneLLM                      | Audio, Point cloud, IMU, Depth                            |
| JAT                         | Vision, Language, Action                                  |
| OpenVLA                     | Vision, Language, Action                                  |
| gpt-4o-2024-05-13           | Vision, Language, Audio                                   |
| claude-3-5-sonnet-20240620  | Vision, Language                                          |
| gemini-1.5-pro-api-0514     | Vision, Language, Audio                                   |
| gemma-2-27b-it              | Language                                                  |
| gemini-1.5-flash-api-0514   | Vision, Language, Audio                                   |
| meta-llama-3-70b-instruct   | Language                                                  |
| mistral-large-2402          | Language                                                  |
| RT-2-X                      | Action                                                    |
| Octo                        | Action                                                    |


## Getting Started

Generate a python environment:

```bash
python -m venv multinet
```

Activate the environment:

```bash
source multinet/bin/activate
```

Finally, install dependencies using:

```bash
pip install .
```
on the directory that has the `toml` file
