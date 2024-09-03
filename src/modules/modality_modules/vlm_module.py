from src.modules.source_modules.openai_module import OpenAIModule

class VLMModule:
    def __init__(self, source):
        source_module = None
        if source == 'openai':
            source_module = OpenAIModule()

        assert source_module is not None, "The source module has not been set correcly. Check required."

    # TODO: Processing the input according to the source module.
    def process_input():
        pass

    # TODO: Processing the output according to the source module.
    def process_output():
        pass

    # TODO: One inference step.
    def infer_step():
        pass
