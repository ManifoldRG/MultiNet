"""
Natural Language Model Interface

Extends the standard ModelInterface for models that produce
natural language action commands instead of discrete action IDs.
"""

from __future__ import annotations

from abc import abstractmethod

try:
    from ..model_interface import ModelInterface, ModelInput, ModelOutput
except ImportError:
    from model_interface import ModelInterface, ModelInput, ModelOutput
from .nl_action_parser import NLActionParser


class NLModelInterface(ModelInterface):
    """
    Model interface for NL-based action prediction.

    Models implementing this interface produce natural language commands
    (e.g., "turn left then move forward") which are parsed to action IDs.

    Subclasses must implement predict_nl() instead of predict().
    """

    def __init__(self):
        self._parser = NLActionParser()

    @abstractmethod
    def predict_nl(self, input: ModelInput) -> str:
        """
        Predict a natural language action command.

        Args:
            input: ModelInput with image and context

        Returns:
            Natural language command string
        """
        ...

    def predict(self, input: ModelInput) -> ModelOutput:
        """
        Predict action by generating NL command and parsing it.

        This wraps predict_nl() for compatibility with the standard
        evaluation harness.
        """
        nl_command = self.predict_nl(input)

        # Parse NL to action sequence; use first action
        # Agent facing defaults to 0 since we don't have it in ModelInput
        actions = self._parser.parse(nl_command, agent_facing=0)
        action = actions[0] if actions else 6

        return ModelOutput(
            action=action,
            reasoning=f"NL command: {nl_command}",
            raw_output=nl_command,
        )
