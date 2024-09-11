import os
from transformers import AutoModelForCausalLM, AutoProcessor
import tensorflow as tf
from PIL import Image
import numpy as np
from openx_dataloader import get_openx_dataloader
from jat_openx_eval import evaluate_jat_model

# Load the model and the processor
model_name_or_path = "jat-project/jat"
processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, ignore_mismatched_sizes=True)

#tfds_path = input('Enter the path to the tfds dataset: ')
#finalds = tf.data.Dataset.load(tfds_path)

#Observe the structure of the TFDS dataset
#print(finalds)

openx_dataset_paths = os.listdir('<path to openx datasets>')

for openx_dataset in openx_dataset_paths:

    tfds_shards = [os.path.join('<path to openx datasets>'+str(openx_dataset), f) for f in os.listdir('<path to openx datasets>'+str(openx_dataset))[:3]]
    tfds_shards = [os.path.join('<path to openx datasets>'+str(openx_dataset), f) for f in os.listdir('<path to openx datasets>'+str(openx_dataset))[:3]]
    #dataloader = get_openx_dataloader(tfds_shards, batch_size=1)

    evaluate_jat_model(model, processor, tfds_shards)


    '''#Iterate through the dataset
    for batch in dataloader:

        model.reset_rl()

        #Because the batch size is 1, 1 batch contains 1 episode, which is why the first element is indexed
        for idx in range(len(batch['continuous_observation'][0])):
       
            elem = batch

            #JAT model generated action

            #print(batch['continuous_observation'][0][idx])

            action = model.get_next_action(processor, text_observation = elem['text_observation'][0][idx], image_observation = elem['image_observation'][0][idx],continuous_observation = elem['continuous_observation'][0][idx], discrete_observation = elem['discrete_observation'][0][idx], reward =elem['reward'][0][idx], action_space=elem['action'][0][idx])
            print('\nModel predicted action')
            print(action)

            print('\nActual action')
            print(elem['action'][0][idx])

            print('#############################################')
   
        
        break'''
    
