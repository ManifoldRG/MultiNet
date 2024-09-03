from src.modules.modality_modules.vlm_module import VLMModule

class OpenXModule:
    def __init__(self, modality: str, source: str) -> None:
        modality_module = None
        if modality == "vlm":
            modality_module = VLMModule(source)

        assert modality_module is not None, "The modality module has not been set correctly. Check required."

    # TODO: Loading the translated data.
    def load_data():
        pass

    # TODO: Split the evaluation set.
    def get_eval_set():
        pass

    # TODO: Evaluation loop.
    def eval_loop():
        pass

    # TODO: Calculating the scores.
    def get_score():
        pass
        