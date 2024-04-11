# MultiNet

We are currently working towards a Multi Modal Generalist Benchmark. The first
version of the project is going to be a pretraining corpus for the
[NEKO](https://github.com/ManifoldRG/Neko) project.


The main idea behind this pretraining corpus is the original NEKO dataset, but
we do not have access to all of the data that Google DeepMind had because most
of it was close sourced. That is why we are using open source alternatives.

The main pretraining corpus of the original GATO project is the following:

![The original gato corpus](./assets/gato_corpus.png)

We currently are starting with evaluating models for the Vision / Language datasets.

This is our complimentary view of what each dataset would be replaced with for the Vision / Language datasets.

| Vision/language dataset | Weight | Replaced by                            | Tokens  |
| ----------------------- | ------ | ---------------------------------------|---------|
| MassiveText             | 6.70%  | PILE                                   | 117.8B  |
| M3W                     | 4.00%  | Multimodal C4                          | 70.34B  |
| ALIGN                   | 0.67%  | COYO-700M                              | 11.78B  |
| MS-COCO                 | 0.67%  | Open Source                            | 11.78B  |
| Conceptual Captions     | 0.67%  | Open Source                            | 11.78B  |
| LTIP                    | 0.67%  | LAION2-B / AESTHETIC-LAION             | 11.78B  |
| OKVQA                   | 0.67%  | AOKVQA (The augmented version of OKVQA)| 11.78B  |
| VQA-V2                  | 0.67%  | Open Source                            | 11.78B  |
| Total Vision + Language | 14.7%  |                                        | 258.82B |

We need a total of 1.5T for the control data and 258.82B for the language and vision data.

That is a total of 1.7T tokens.

We are benchmarking the following models.


| Name         | Inputs                                             | Outputs                             | Open Source | Architecture                                              | 
| ------------ | -------------------------------------------------- | ----------------------------------- | ----------- | ----------------------------------------------------------| 
| LlaVa        | text + image                                       | text                                | Yes         | Vision Encoder (CLIP ViT-L/14) + Vicuna                   | 
| PandaGPT     | text + image/video + audio + depth + thermal + IMU | text                                | Yes         | Multimodal encoders from ImageBind + Vicuna               | 
| Mini-GPT4    | text + image                                       | text                                | Yes         | Q-Former & ViT + Vicuna                                   | 
| NExT-GPT     | text + image + audio + video                       | text + image + audio + video        | Yes         | Multimodal encoders from ImageBind + Vicuna               | 
| AnyGPT       | text + image + audio                               | text + image + audio (speech/music) | Not yet     | SEED tokenizer + SpeechTokenizer + Encodec + LLaMA-2 7B   | 
| IDEFICS      | images + video + text                              | text                                | Yes         | OpenClip + LlaMA                                          | 
| OpenFlamingo | images + text                                      | text                                | Yes         | CLIP ViT-L/14 + MPT-1B / RedPajama3B / MPT-7B             | 



## Getting Started

Generate a python environment:

```bash
python -m venv multinet
```
