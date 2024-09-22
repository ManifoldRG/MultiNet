# Dataset definitions

Any models that have not been explicitly trained on the control tasks requires the proper instruction prompt to perform RL(Reinforcement Learning) tasks properly. To do this, we define the necessary information for each dataset in this directory.

<br/>

---

### Category

- **Descriptions**: These are descriptions of each dataset and environment. Any required information to understand a certain environment (e.g. the objects in the observation, the goal of the task, the restriction when performing a task, etc.) should be defined. (**Required**)
- **Action spaces**: The action space helps the model to understand what kind of actions are allowed during the task. Each action has own index and it can be either continuous or discrete. If it is discrete, it can have a mapping from the value of an option and the description of that option. If it is continuous, it has the range of values allowed. (**Required**)
- **Action exclusiveness**: These indicate whether the action output in each environment should be exclusive or not. If it is, that means the model should only one action at a time. (e.g. Atari) If it isn't, the action values can be multiple. (e.g. any robotics tasks) (**Required**)
- **Additional instructions**: If we need to inject any additional information into the model's prompt, we define it here. This is not mandatory and some datasets don't have to have these.

<br/>

---

### Format

First, make a file named `{dataset}.py`. (e.g. `openx.py`, `atari.py`, etc.) Then import this file in each dataset module file to use the definitions inside of the file as constants. (See line 4 in `src/modules/dataset_modules/openx_module.py` for example.)

*Note: OpenX Consists of another multiple datasets. Thus, keep in mind that the definitions in `openx.py` has **one more** depths compared to those in other datasets, where the key is the name of sub-dataset and the value is actual definition.*

- **Descriptions**

  ```python
  DESCRIPTIONs = {
      {ENVIRONMENT_NAME}: [
          {sentence1},
          {sentence2},
          ...
      ],
      ...
  }
  ```

  `DESCRIPTIONS` is a dictionary which has the environment's name as a key and the list of strings as a value. Each string is just a sentence in the description.

  *Note that each dataset can have various environments in it. In OpenX's case, `text_observation` works as a name of the environment. Make sure to research the specifications of the dataset carefully.*

- **Action spaces**

  ```python
  ACTION_SPACES = {
      {ENVIRONMENT_NAME}: {
          0: (...),
          1: (...),
          ...
      },
      ...
  }
  ```

  `ACTION_SPACES` is a dictionary which has the environment's name as a key and another dictionary as a value. A sub-dictionary has a key, which is the index of each action, and a value, which is a tuple. Each tuple has a different format depending on the property of the action.

  - Discrete action: In this case, the tuple's length is 2. The first element is the definition of the action. The second element is another dictionary, where the key is the actual value of an option and the value is the description of that option.
  - Continuous action: In this case, the tuple's length is 3. The first element is the definition of the action. The second element is the minimum value of that action. The third element is the maximum value of the action.

- **Action exclusiveness**

  ```python
  ACTION_EXCLUSIVENESS = {
      {ENVIRONMENT_NAME}: True or False,
      ...
  }
  ```

  `ACTION_EXCLUSIVENESS` is a dictionary which has the environment's name as a key and a boolean value as a value. If the model should generate only one action at a time, this should be set into `True`. Otherwise, it is set to `False`.

  - `True`: The model will generate the output as `{ACTION_INDEX} {OPTION_INDEX}` (discrete) or `{ACTION_INDEX} {CONTINUOUS_VALUE}` (continuous).
  - `False`: The model will generate the output as a vector which contains all corresponding values for all actions.

- **Additional instructions**

  ```python
  ADDITIONAL_INSTRUCTIONS = {
      {ENVIRONMENT_NAME}: [
          {sentence1},
          {sentence2},
          ...
      ],
      ...
  }
  ```

  `ADDITIONAL_INSTRUCTIONS` has the same format as `DESCRIPTIONS`.

<br/>

****