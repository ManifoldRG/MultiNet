import torch
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

# Import the custom action tokenizer
from action_tokenizer import ActionTokenizer

def load_model_and_processor():
    """
    Loads the Magma-8B model with 4-bit quantization and its associated processor.
    """
    print("🚀 Loading Magma-8B model and processor...")

    # Configuration for 4-bit quantization to reduce memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Magma-8B",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config
    )

    processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
    
    print("✅ Model and processor loaded successfully.")
    return model, processor

def get_fractal_dataset_iterator():
    """
    Loads and prepares an iterator for the Fractal 2022 dataset.
    """
    print("📂 Loading Fractal dataset iterator...")
    
    def dataset2path(dataset_name):
        return f'gs://gresearch/robotics/{dataset_name}/0.1.0'

    dataset_name = 'fractal20220817_data'
    builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
    ds = builder.as_dataset(split='train')
    
    print("✅ Dataset iterator ready.")
    return iter(ds)

def main():
    """
    Main function to run the inference pipeline.
    """
    model, processor = load_model_and_processor()
    episode_iterator = get_fractal_dataset_iterator()
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    dtype = torch.float16
    MAX_EPISODES_TO_CHECK = 100
    found_and_processed = False

    for i in range(MAX_EPISODES_TO_CHECK):
        print(f"\nSearching for valid sample in episode #{i+1}...")
        try:
            episode = next(episode_iterator)
        except StopIteration:
            print("Reached the end of the dataset.")
            break

        first_step = next(iter(episode['steps']))
        image = None
        language_instruction = None

        # Extract image
        if 'observation' in first_step and 'image' in first_step['observation']:
            image_tensor = first_step['observation']['image']
            image = Image.fromarray(image_tensor.numpy())
        else:
            continue

        # Extract language instruction
        if 'observation' in first_step and 'natural_language_instruction' in first_step['observation']:
            instruction_tensor = first_step['observation']['natural_language_instruction']
            if tf.is_tensor(instruction_tensor) and tf.size(instruction_tensor) > 0:
                language_instruction = instruction_tensor.numpy().decode('utf-8').strip()

        # If both image and instruction are found, run inference
        if image and language_instruction:
            print(f"Found valid sample with instruction: '{language_instruction}'")
            print("🤖 Running MAGMA inference...")

            convs = [
                {"role": "system", "content": "You are agent that can see, talk and act."},
                {"role": "user", "content": f"<image_start><image><image_end>\\n{language_instruction}"},
            ]

            prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
            inputs = processor(images=[image], texts=prompt, return_tensors="pt")
            
            # Prepare inputs for the model
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
            inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)

            for key, tensor in inputs.items():
                inputs[key] = tensor.to("cuda").to(dtype) if key == 'pixel_values' else tensor.to("cuda")

            generation_args = {"max_new_tokens": 128, "temperature": 0.0, "do_sample": False, "use_cache": False, "num_beams": 1}

            with torch.inference_mode():
                generate_ids = model.generate(**inputs, **generation_args)

            # Decode the output to get the robot action
            output_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
            output_ids_np = output_ids.cpu().numpy()
            robot_action_7dof = action_tokenizer.decode_token_ids_to_actions(output_ids_np[0])

            print("\n" + "="*50)
            print(" MAGMA Model Response (7-DoF Action):")
            print(robot_action_7dof)
            print("="*50)

            found_and_processed = True
            break # Stop after the first successful run

    if not found_and_processed:
        print(f"\nCould not find a valid episode to process after checking {MAX_EPISODES_TO_CHECK} episodes.")

if __name__ == "__main__":
    main()