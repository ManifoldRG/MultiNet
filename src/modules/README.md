# GenESIS: Generalizable Extendable Stratified Inference System

MultiNet project uses the framework called **GenESIS**: **Gen**eralizable and **E**xtendable **S**tratified **I**nference **S**ystem to adapt a wide range of models into multiple types of tasks or datasets for scaling effectively while reducing the engineering efforts as much as possible. The core insights of GenESIS are 1) <u>Interchangeability</u>: Any models or datasets should interchangeably support each other, 2) <u>Abstraction</u>: Each module should share the same architecture and logic to enable the user to understand and test each module easily, and 3) <u>Encapsulation</u>: The programmer does not have to know the details on other modules and is allowed to concentrate on the models or datasets that are targeted. In this way, any models or datasets can easily be added to the MultiNet benchmark without affecting the existing implementations.

<img src="../../assets/framework-figure.png" alt="The figure of the inference framework."/>

<br/>

## Example Usecase: Mapping LLMs to Robot Actions in MultiNet

For example, in MultiNet v0, GenESIS is used to evaluate GPT-4-o on the OpenX datasets. Since the GPT-4 is a vision-language model for general-purpose tasks, it is necessary to set up a proper instruction prompt, pre-processing of the input data in each dataset into a form that GPT-4 can consume, the management of chat history list for correct API calls, and conversion of generated text outputs from the model to compare them with the action labels. This results in a high implementation effort and even if it is successfully implemented, most of the codes should inevitably be hard-coded, which degrades the readability of codes and increases the complexity of future implementations whenever any models or datasets are added. GenESIS modularizes and abstracts the processes to separate each dataset/model-specific part and maximize the code reusability as much as possible, eventually aiming to help the scaling of this huge MultiNet system. For V0, most models supported by OpenAI API are ready to run on all OpenX datasets based on GenESIS.

<br/>

---

## Modules

GenESIS consists of three components called "modules". The relation between each module is hierarchical (that's why it is "stratified") and as we go from the upper module to the lower module, the more general and shared the features become. The categories of the modules are as follows:

1. **Dataset Module**: This module is the parent module to use a modality module and source module. This module includes the dataset-specific features and all logics that should be performed for a specific dataset or task are implemented in this module. The dataset module can have the following responsibilities (but not limited to):

   - It loads the dataloader object and fetches a batch that has the inputs and labels. Since each dataloader is likely to be different across the dataset, communicating with the dataloader is suitable job for the dataset module.
   - It generates a system instruction prompt if it uses an LLM or VLM module. Since those modules require a proper instruction prompt and a prompt is based on the dataset-specific definitions, it is the responsibility of the dataset module.
   - It compares the outputs from a model and the labels to calculate the evaluation metrics. Since these metrics are different for each dataset or task, this calculation is implemented inside of the dataset module.

   For example, the OpenXModule uses the pre-implemented OpenXDataloader to process the batch and calculates MSE scores and success rates after collecting the action outputs from the model. Also, it generates the system prompt and passes it to the source module, which is used if the source module is either the VLMModule or LLMModule.

2. **Modality Module**: This module connects the dataset and source modules to smoothly process the data in any modalities. Each modality module can handle different modalities (one or more), such as images, texts, arrays, etc. The modality module can have the following responsibilities (but not limited to):

   - It takes the batch from the dataset module and converts any data in it into the types it can handle. Also, it puts a tag to each data so that the source module can understand it.
   - It takes the output from the model and converts it into its original type so that the dataset module can properly evaluate the result.

   For example, the VLMModule processes two data types: images and texts. If there are images or texts in the batch, it just passes them as are. On the other hands, if it encounters with any other types of data, it converts them into string type before passing them to the source module. After the AI model in the source module generates the output, it converts the output into the original form before passing it to the dataset module.

3. **Source Module**: This module uses the actual AI model to generate the output. Depending on the model, it can have the model object itself or the client for API calls to communicate with the close-source models. The source module can have the following responsibilities (but not limited to):

   - It executes the inference step to generate the output from the model.
   - It manages the multi-turn context. It updates the context memory when any input is given or after the model generates the output. The context memory is kept consistent until the modality module wants it to be reset. Also, it is the source module's responsibility to make sure that the total input size fits into the maximum context window of the model.
   - It has the tokenizer or encoder to process the given input data into the final format that the model can take. These are also be used for calculating the number of tokens or input size.

   For example, the OpenAIModule has the OpenAI's AI Client to get the output from any OpenAI's AI models. Since most of the OpenAI models are LLMs or VLMs, it assumes that all inputs it gets are either images or texts. It also has the mechanism to calculate the number of text/image tokens and a truncation algorithm to fit the total input size into the context size of the model it uses.

<br>

---

## Definitions

These are the main variables and functions that each module should support for GenESIS to work properly end-to-end. Note that these are minimally required functions and you can implement additional helper functions or variables on your own depending on the designs or tasks.

($B$: Batch size, $K$: # of few-shot examples, $N$: # of data that consists of one input)

1. **Dataset Module**

   - Required variables

     - `self.modality_module (Object)`: The modality module object to be accessed.
     - `self.batch_size (int)`: The batch size to be used for inference.
     - `self.k_shots (int)`: The number of few-shot examples for each inference.

   - Required functions

     - ```python
       def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_size: int, k_shots: int) -> None
       ```

       - `__init__` is the constructor.
       - Inputs
         - `disk_root_dir`: The root directory that contains the translated TFDS files. The dataset module assumes that these files are already set up as the shards to be loaded via `tensorflow.data.Dataset`. Depending on your need, you can implement the dataloader to load the batches more effectively.
         - `modality`: The name of the modality module. This is used for initializing the modality module.
         - `source`: The name of the source module. This is used for initializing the modality module.
         - `model`: The name of the AI model. This is used for initializing the modality module.
         - `batch_size`: The batch size used for inference.
         - `k_shots`: The number of few-shot examples to be included during the evaluation.
       - Outputs: N/A.

     - ```Python
       def run_eval(self) -> None
       ```
     
       - `run_eval` is the main function to run the total inference steps and calculate the scores.
     
         - This function is called from `src/eval/eval_main.py` to evaluate the model on one dataset in Multinet.
     
         - It should have the logic to load the dataloader (or translated data files), iterate the batches, call the modality module (and source module in it) to run the inference steps, and compare the answers and generated outputs to get the evaluation scores.
     
       - Inputs: N/A.
     
       - Outputs: N/A.
     
         - Note that this is the main function, you can output whatever you want without returning anything. You can just print out the scores or you can save a file that contains the results.
     

2. **Modality Module**

   - Required variables

     - `self.source_module (Object)`: The source module object to be accessed.

   - Required functions

     - ```python
       def __init__(self, source: str, model: str) -> None
       ```
     
       - `__init__` is the constructor.
       - Inputs
         - `source`: The name of the source module. This is used for initializing the source module.
         - `model`: The name of the AI model. This is used for initializing the source module.
       - Outputs: N/A.
     
     - ```python
       def infer_step(self, 
                      cur_inputs: list[list[tuple[str, Any]]], 
                      k_shots_examples: list[list[tuple[str, list[tuple[str, Any]]]]]=[],
                      instructions: list[str]=[],
                      output_types: list[type]=[]
                   ) -> list[Any]
       ```
     
       - `infer_step` is the function that runs one inference step using the source module.
         - This function should be called from `run_eval` function in the dataset module.
         - It should call `infer_step` function to get the actual model outputs from the source module.
       - Inputs
         - `cur_inputs`: The current inputs that the model should infer to get the outputs. $(B, N, 2, *)$
           - The reason why one input is a sequence of data is that some models require multiple inputs in one step. For example, a VLM often uses images and texts to generate one output.
           - Each data set is a tuple, with the first element being a string description (e.g., "image_observation", "robot_state") and the second element being the actual data. The description helps the source model understand what each data represents. It is up to the source module to decide whether to include this description in the model.
         - `k_shots_examples`: The context examples for few-shot learning. $(B, K, *, 2, *, 2, *)$
           - Each $k$-shot list has the tuples with the first element is either "input" or "output". Typically, the length of one $k$-shot example is $2$, since there is one input sequence and one output.
           - Each input sequence is a list of input data similar with one input in `cur_inputs`.
         - `instructions`: The system prompt instructions for inference. $(B)$
         - `output_types`: The output types that the modality module should give to the dataset module.
       - Outputs
         1. The generated outputs from the model. $(B, *)$
            - Note that the data type in this list should align with `output_types` given as an input.
     

3. **Source Module**

   - Required variables

     - `self.model (str or Object)`: The AI model to use.
       - Note that some source modules use the API calls. In this case, this variable is just a string without initializing the model object specifically. (Refer to `OpenAIModule`)

   - Required functions

     - ```python
       def __init__(self, model: str) -> None
       ```

       - `__init__` is the constructor.
       - Inputs
         - `model`: The name of the AI model. This is used for initializing the actual model object or definition.
       - Outputs: N/A.
   
     - ```python
       def infer_step(self, inputs: list[list[tuple[str, Any]]], system_prompts: list=[]) -> list[Any]
       ```
   
       - `infer_step` is the function that runs one inference step using the model.
       - Inputs
         - `inputs`: The current inputs that the model should infer to get the outputs. $(B, N, 2, *)$
           - Like `cur_inputs` in the modality module, each data is tagged with the description. This description can be freely used for implementing the pre-processing logic before the inputs are put into the model.
         - `system_prompts`: The list of system instruction prompts if any model requires it. $(B)$
       - Outputs
         1. The generated outputs from the model. $(B, *)$
   
     - ```python
       def add_data(self, type: str, data: list[list[tuple[str, Any]]]) -> None
       ```
   
       - `add_data` is the function that adds the new batch into the context history inside of the source module.
         - We recommend you to assume that `add_data` should be called only when the additional data is put into the context before performing the actual inference. For example, an inference with the few-shot demonstrations could use this function. (Refer to `VLMModule`)
         - `infer_step` should call this function to update the current input and generated output into the context history by default.
       - Inputs
         - `type`: This indicates whether the data to put are `input` or `output`. This can help the source module to tag each data correctly.
         - `inputs`: The inputs to be put into the context history. $(B, N, 2, *)$
       - Outputs: N/A.
   
     - ```python
       def clear_history(self) -> None
       ```
   
       - `clear_history` is the function that clears all context history in the source module.
       - Inputs: N/A.
       - Outputs: N/A.
   
   *Note that in MultiNet V0, we only have `OpenAIModule` as a source module. For the simplicity of implementation, `OpenAIModule` assumes that the data it gets is not a batch, not a single data. So the current implementation might have different data shapes. However, other source modules that use other models should be able to process the batch. Refer to the [issue](https://github.com/ManifoldRG/MultiNet/issues/215).*

<br/>

---
