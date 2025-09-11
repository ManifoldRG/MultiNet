import tensorflow as tf
import argparse
import os
from PIL import Image

def analyze_dataset(shard_path: str, num_steps: int = 5):
    """Loads a single shard and analyzes its content."""
    if not os.path.exists(shard_path):
        print(f"Error: Shard path not found at {shard_path}")
        return

    print(f"Loading shard: {shard_path}")
    try:
        dataset = tf.data.Dataset.load(shard_path)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("\n--- Analyzing first {num_steps} steps ---")
    for i, step in enumerate(dataset.take(num_steps)):
        print(f"\n--- Step {i+1} ---")

        # Analyze observation
        observation = step.get('observation', {})
        image = observation.get('image')
        instruction = observation.get('natural_language_instruction')
        state = observation.get('state')

        if image is not None:
            try:
                img_array = image.numpy()
                pil_img = Image.fromarray(img_array)
                img_path = f"/tmp/observation_image_{i+1}.png"
                pil_img.save(img_path)
                print(f"  - Saved observation image to: {img_path}")
            except Exception as e:
                print(f"  - Could not save image: {e}")
        else:
            print("  - No image in observation.")

        if instruction is not None:
            print(f"  - Instruction: {instruction.numpy().decode('utf-8')}")
        else:
            print("  - No instruction.")

        if state is not None:
            print(f"  - State vector: {state.numpy()}")
        else:
            print("  - No state vector.")

        # Analyze action
        action = step.get('action', {})
        print("  - Action structure:")
        for key, value in action.items():
            print(f"    - {key}: {value.numpy()}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a single processed OpenX shard.")
    parser.add_argument(
        "shard_path",
        type=str,
        help="The absolute path to the dataset shard directory (e.g., .../translated_shard_0)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of steps to analyze from the shard."
    )
    args = parser.parse_args()
    analyze_dataset(args.shard_path, args.num_steps)

if __name__ == "__main__":
    main()
