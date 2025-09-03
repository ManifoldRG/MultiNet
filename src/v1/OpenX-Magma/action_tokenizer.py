from typing import List, Union
import numpy as np
from transformers import PreTrainedTokenizerBase

class ActionTokenizer:
    """
    A tokenizer to convert continuous robot actions into discrete tokens for a language model
    and decode token IDs back into continuous actions.
    """
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Initializes the ActionTokenizer.

        Args:
            tokenizer (PreTrainedTokenizerBase): The base tokenizer from the Hugging Face model.
            bins (int): The number of discrete bins to discretize the action space into.
            min_action (int): The minimum value of the continuous action space.
            max_action (int): The maximum value of the continuous action space.
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """
        Decodes discretized action indices into their string representation.
        """
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def encode_actions_to_token_ids(self, action: np.ndarray) -> np.ndarray:
        """
        Encodes continuous actions into the model's token ID space.
        """
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        return self.tokenizer.vocab_size - discretized_action

    def encode_actions_to_discrete_ids(self, action: np.ndarray) -> np.ndarray:
        """
        Encodes continuous actions into discrete bin IDs (0 to n_bins-1).
        """
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        return discretized_action

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Decodes token IDs from the model's output back into continuous action values.
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        """
        Returns the number of bins in the action space.
        """
        return self.n_bins