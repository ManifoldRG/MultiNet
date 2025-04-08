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
    
    # Dictionary to store mappings
    mappings = {}
    valid_count = 0
    min_action = float('inf')
    max_action = float('-inf')
    unique_integer_ranges = Counter()
    
    print("\nTrying all possible action tokens...")
    
    # Since we're working with action_horizon=1 and action_dim=1,
    # we only need to try single tokens that would be valid after:
    # 1. PaliGemma decode -> encode -> act_tokens_to_paligemma_tokens -> FAST decode
    
    # Try a range of tokens that would be valid after the mapping
    # paligemma_token = paligemma_vocab_size - 1 - 128 - fast_token
    # We know paligemma_vocab_size is around 256000-258000
    # 257152
    for token_id in range(300000):  # PaliGemma token range id 254980 to 257023. 1078 tokens
        try:

            action = tokenizer.extract_actions(np.array([4022, 235292, 235248, token_id]), action_horizon=ACTION_HORIZON, action_dim=ACTION_DIM)
            if action is None:
                continue
                
            if action.tolist()[0][0] == 0:
                print(f"skipping {token_id} action: {action}")
                continue
        
            # Store mapping
            mappings[token_id] = {
                'paligemma_token': int(token_id),
                'action_value': round(float(action.tolist()[0][0]), 1)  # Ensure it's JSON serializable
            }
            unique_integer_ranges[int(action.tolist()[0][0])] += 1
            # Update statistics
            valid_count += 1
            min_action = min(min_action, round(float(action.tolist()[0][0]), 1))
            max_action = max(max_action, round(float(action.tolist()[0][0]), 1))
        
            print(f"Processed up to token {token_id}, found {valid_count} valid tokens")
            
        except Exception as e:
            continue
    
    # Print summary
    if valid_count > 0:
        print(f"\nFirst few mappings:")
        for token_id in sorted(mappings.keys())[:5]:
            print(f"PaliGemma Token {token_id} -> Action Value {mappings[token_id]['action_value']}: {mappings[token_id]}")
            
        print(f"\nLast few mappings:")
        for token_id in sorted(mappings.keys())[-5:]:
            print(f"PaliGemma Token {token_id} -> Action Value {mappings[token_id]['action_value']}: {mappings[token_id]}")
            
        print(f"\nValid PaliGemma token range: [{min(mappings.keys())}, {max(mappings.keys())}]")
        print(f"Action value range: [{min_action}, {max_action}]")
        print(f"Total valid mappings: {valid_count}")
        
        # Save mappings
        with open('src/eval/profiling/openpi/scripts/bpe_token_to_action_value_mappings.json', 'w') as f:
            json.dump({
                'mappings': mappings,
                'stats': {
                    'valid_token_range': {
                        'min': min(mappings.keys()),
                        'max': max(mappings.keys())
                    },
                    'action_value_range': {
                        'min': round(min_action, 1),
                        'max': round(max_action, 1)
                    },
                    'unique_integer_ranges': dict(unique_integer_ranges),
                    'total_valid_mappings': valid_count
                }
            }, f, indent=2)
        print("\nMappings saved to bpe_token_to_action_value_mappings.json")
    else:
        print("\nNo valid mappings found!")

if __name__ == "__main__":
    map_bpe_to_actions()
