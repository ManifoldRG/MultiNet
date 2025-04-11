import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

import json
from src.eval.profiling.openpi.src.openpi.models.tokenizer import FASTTokenizer
import numpy as np
from collections import Counter

# Constants from processor_config.json
MIN_TOKEN = -354
SCALE = 10
ACTION_HORIZON = 1
ACTION_DIM = 1
FAST_SKIP_TOKENS = 128

def map_bpe_to_actions():
    tokenizer = FASTTokenizer()
    
    token_to_actions = {}  # Modified to store action value for each token
    unique_action_values = set()
    valid_count = 0
    min_action = float('inf')
    max_action = float('-inf')
    
    print("\nTrying all possible action tokens...")
    
    for token_id in range(300000):  # PaliGemma token range id 254980 to 257023. 1078 tokens
        try:
            action = tokenizer.extract_actions(np.array([4022, 235292, 235248, token_id]).astype(np.int32), 
                                            action_horizon=ACTION_HORIZON, 
                                            action_dim=ACTION_DIM)
            action_value = action.tolist()[0][0]
            if action is None or action_value == 0:
                print(f"skipping {token_id} action: {action_value}")
                continue

            # Store token_id -> action_value mapping
            token_to_actions[str(token_id)] = action_value
            unique_action_values.add(action_value)
            
            # Update statistics
            valid_count += 1
            min_action = min(min_action, action_value)
            max_action = max(max_action, action_value)
            
        except Exception as e:
            continue
    
    # Print summary
    if valid_count > 0:
        print(f"\nAction value range: [{min_action}, {max_action}]")
        print(f"Total valid mappings: {valid_count}")
        print(f"Unique action values: {len(unique_action_values)}")

        # Find tokens that map to the same action value
        action_to_tokens = {}
        for token, action in token_to_actions.items():
            if action not in action_to_tokens:
                action_to_tokens[action] = []
            action_to_tokens[action].append(int(token))

        # Print statistics about duplicates
        print("\nAction values with multiple tokens:")
        for action_val, tokens in action_to_tokens.items():
            if len(tokens) > 1:
                print(f"Action {action_val}: {len(tokens)} tokens -> {tokens}")

        # Save mappings
        with open('src/eval/profiling/openpi/scripts/bpe_token_to_action_value_mappings.json', 'w') as f:
            json.dump({
                'mappings': token_to_actions,  # Save token -> action mapping
                'duplicate_stats': {str(action): len(tokens) for action, tokens in action_to_tokens.items()},
                'stats': {
                    'action_value_range': {
                        'min': min_action,
                        'max': max_action
                    },
                    'total_valid_mappings': valid_count,
                    'unique_action_values': len(unique_action_values)
                }
            }, f, indent=2)
        print("\nMappings saved to src/eval/profiling/openpi/scripts/bpe_token_to_action_value_mappings.json")
    else:
        print("\nNo valid mappings found!")

if __name__ == "__main__":
    map_bpe_to_actions()
