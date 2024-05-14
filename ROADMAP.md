# ROADMAP 

## Language and Vision
For the Images and Text, we can do the following.

| Vision/language dataset | Weight | Replaced by                            | Tokens  | Links |
| ----------------------- | ------ | ---------------------------------------|---------|-------|
| MassiveText             | 6.70%  | PILE                                   | 117.8B  | https://huggingface.co/datasets/EleutherAI/pile |
| M3W                     | 4.00%  | OBELICS                                | 70.34B  | https://huggingface.co/datasets/HuggingFaceM4/OBELICS |
| ALIGN                   | 0.67%  | COYO-700M                              | 11.78B  | https://huggingface.co/datasets/kakaobrain/coyo-700m|
| MS-COCO                 | 0.67%  | Open Source                            | 11.78B  | https://huggingface.co/datasets/shunk031/MSCOCO |
| Conceptual Captions     | 0.67%  | Open Source                            | 11.78B  | https://huggingface.co/datasets/conceptual_captions
| LTIP                    | 0.67%  | LAION2-B / AESTHETIC-LAION  /WORST CASE SCENARIO COYO700M            | 11.78B  | https://huggingface.co/datasets/kakaobrain/coyo-700m|
| OKVQA                   | 0.67%  | AOKVQA (The augmented version of OKVQA)| 11.78B  | https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA|
| VQA-V2                  | 0.67%  | Open Source                            | 11.78B  | https://huggingface.co/datasets/HuggingFaceM4/VQAv2|
| Total Vision + Language | 14.7%  |                                        | 258.82B | |

We need a total of 1.5T for the control data and 258.82B for the language and vision data.

The following is what we can do.

We have to look a bit more into the pile, maybe use another thing
because it is currently in a tough spot.

We have to change Multimodal C4 because it does not have enough tokens. 
We should change to OBELICS. And sample it.

AESTHETIC-LAION or LAION2-B, it seems that both of them have been taken down. We could 
replace them and merge both 0.67% to COYO-700M.

- MS-COCO we should just sample it
- COYO-700M we should just sample it
- Conceptual Captions, just sample it.
- AOKVQA just sample it.
- VQA-V2 just sample it.

After doing this, we can test it as they tested the PILE, just checking the entropy
and the results of the different models. Let's remember that the pile is just 22 different
datasets into one.

## Control

This is where the rubber hits the road. We have to replicate some of this.

| Environment                   | Tasks | Episodes | Approx. Tokens | Weight | Replaced by | Status    | Links |
| ----------------------------- | ----- | -------- | -------------- | ------ | ----------- |-----------|-------|
| DM Lab                        | 254   | 16.4M    | 194B           | 9.35%  |RL Unplugged | IMPLEMENT | https://www.samplefactory.dev/09-environment-integrations/dmlab  |
| ALE Atari                     | 51    | 63.4K    | 1.26B          | 9.50%  |JAT-dataset  | DONE      | https://huggingface.co/datasets/jat-project/jat-dataset |
| ALE Atari Extended            | 28    | 28.4K    | 565M           | 10.00% |JAT-dataset  | DONE      | https://huggingface.co/datasets/jat-project/jat-dataset |
| Sokoban                       | 1     | 27.2K    | 298M           | 1.33%  |IMPLEMENT    | NOTHING   | https://github.com/mpSchrader/gym-sokoban |
| Baby AI                       | 46    | 4.61M    | 22.8B          | 9.06%  |JAT-dataset  | JUST RUN  | https://huggingface.co/datasets/jat-project/jat-dataset |
| DM Control Suite              | 30    | 395K     | 22.5B          | 4.62%  |D4RL         | DONE?     | https://github.com/Farama-Foundation/D4RL |
| DM Control Suite Pixels       | 28    | 485K     | 35.5B          | 7.07%  |V-D4RL       | DONE?     | https://github.com/conglu1997/v-d4rl |
| DM Control Suite Random Small | 26    | 10.6M    | 313B           | 3.04%  |D4RL         | DONE?     | https://github.com/Farama-Foundation/D4RL |
| DM Control Suite Random Large | 26    | 26.1M    | 791B           | 3.04%  |D4RL         | DONE?     | https://github.com/Farama-Foundation/D4RL |
| Meta-World                    | 45    | 94.6K    | 3.39B          | 8.96%  |JAT-dataset  | DONE      | https://huggingface.co/datasets/jat-project/jat-dataset |
| Procgen Benchmark             | 16    | 1.6M     | 4.46B          | 5.34%  |Procgen      | TRANSLATE | https://github.com/facebookresearch/gen_dgrl |
| RGB Stacking simulator        | 1     | 387K     | 24.4B          | 1.33%  |IMPLEMENT    | NOTHING   | https://github.com/google-deepmind/rgb_stacking
| RGB Stacking real robot       | 1     | 15.7K    | 980M           | 1.33%  |OPEN-X       | NOTHING   | https://huggingface.co/datasets/jxu124/OpenX-Embodiment |
| Modular RL                    | 38    | 843K     | 69.6B          | 8.23%  |LocoMuJoCo   | NOTHING      | https://github.com/ManifoldRG/NEKO |
| DM Manipulation Playground    | 4     | 286K     | 6.58B          | 1.68%  |LocoMuJoCo   | DONE      | https://github.com/ManifoldRG/NEKO |
| Playroom                      | 1     | 829K     | 118B           | 1.33%  |IMPLEMENT    | NOTHING   | I can not find where we could get it |
| Total                         | 596   | 63M      | 1.5T           | 85.21% |             |           |

### TRANSLATE

This should be the easy ones, start with this.

### DONE?

Translate V-D4RL and the rest are supposed to be at D4RL. 

### NOTHING

We have to make this from scratch, it is normally not that difficult but we need the compute.
