import sys
import os
# add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_utils.piqa_dataloader import get_piqa_test_dataloader
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

_, dataloader = get_piqa_test_dataloader("../processed_datasets/piqa/test", batch_size=1)
# get one sample from the dataloader
sample = next(iter(dataloader))
# create a dummy image tensor
dummy_image = torch.zeros(3, 224, 224)

inputs = processor(text=sample["question"][0], images=dummy_image,
                  padding="longest", do_convert_rgb=True, return_tensors="pt").to("cuda")
model.to(device)
inputs = inputs.to(dtype=model.dtype)

with torch.no_grad():
  output = model.generate(**inputs, max_length=496)
print("answered: \n")
print(processor.decode(output[0], skip_special_tokens=True))