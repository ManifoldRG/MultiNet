# GenESIS: Generalizable Extendable Stratified Inference System

MultiNet project uses the framework called **GenESIS**: **Gen**eralizable and **E**xtendable **S**tratified **I**nference **S**ystem to adapt a wide range of models into multiple types of tasks or datasets for scaling effectively while reducing the engineering efforts as much as possible. The core insights of GenESIS are 1) <u>Interchangeability</u>: Any models or datasets should interchangeably support each other, 2) <u>Abstraction</u>: Each module should share the same architecture and logic, and 3) <u>Encapsulation</u>: The programmer does not have to know the details on other modules and is allowed to concentrate on the models or datasets that are targeted. In this way, any models or datasets can easily be added to the MultiNet benchmark without affecting the existing implementations.

<img src="../../assets/framework-figure.png" alt="The figure of the inference framework."/>

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

1. **Dataset Module**
   - Required variables
     - `self.modality_module (Object)`: The modality module object to be accessed.
     - `self.batch_size (int)`: The batch size to be used for inference.
     - `self.k_shots (int)`: The number of few-shot examples for each inference.
   - Required functions
     - `def run_eval(self) -> None`: The main function to run the total inference step and calculate the scores.
       - This function is called from `src/eval/eval_main.py` to evaluate the model on one dataset in Multinet.
       - It should have the logic to load the dataloader, iterate through the loader to get the batch, call the modality module (and source module in it) to run the inference steps, and compare the answers and generated outputs to get the evaluation scores.
2. **Modality Module**
   - Required variables
     - 
3. **Source Module**

<br/>

---
