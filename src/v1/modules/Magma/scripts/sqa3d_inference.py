import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field, fields
from pathlib import Path, types
from typing import Dict, Any, List, Union, Tuple
import re
import numpy as np
import torch
from PIL import Image
import gc
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.modeling_outputs import ModelOutput
from sentence_transformers import SentenceTransformer, util


# Import SQA3D-specific modules
from src.data_utils.sqa3d_dataloader import get_sqa3d_dataloader
from definitions.sqa3d_prompt import SQA3DDefinitions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PatchedMagmaCausalLMOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor] = None
    attentions: Tuple[torch.FloatTensor] = None

def validate_text_output(output: Any) -> bool:
    if output is None:
        return False
    if isinstance(output, str) and len(output.strip()) > 0:
        return True
    return False

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def validate_outputs_and_calculate_metrics(similarity_model: SentenceTransformer, outputs: List[str], labels: List[str]):
    exact_matches = []
    similarity_scores = []
    total_invalid_preds = 0
    normalized_preds = []

    for i, output in enumerate(outputs):
        if validate_text_output(output):
            # Normalize both output and label for fair comparison
            normalized_output = normalize_text(output)
            normalized_label = normalize_text(labels[i])

            # Calculate exact match
            exact_match = 1.0 if normalized_output == normalized_label else 0.0
            exact_matches.append(exact_match)

            # Calculate similarity score
            emb1 = similarity_model.encode(output, convert_to_tensor=True)
            emb2 = similarity_model.encode(labels[i], convert_to_tensor=True)

            similarity = util.cos_sim(emb1, emb2).item()
            similarity_scores.append(similarity)

            normalized_preds.append(normalized_output)
        else:
            # Invalid output - assign worst possible scores
            exact_matches.append(0.0)
            similarity_scores.append(0.0)
            total_invalid_preds += 1
            normalized_preds.append("")

    return exact_matches, similarity_scores, total_invalid_preds, normalized_preds

def calculate_final_metrics(exact_matches: List[float], similarity_scores: List[float], total_invalid_preds: int) -> Dict[str, Any]:
    result = {}

    # Calculate accuracy metrics
    total_samples = len(exact_matches)
    exact_match_accuracy = sum(exact_matches) / total_samples if total_samples > 0 else 0.0
    exact_match_accuracy_without_invalids = sum(exact_matches) / (total_samples - total_invalid_preds) if total_samples - total_invalid_preds > 0 else 0.0

    # Calculate similarity metrics
    avg_similarity_score = sum(similarity_scores) / total_samples if total_samples > 0 else 0.0
    max_similarity_score = max(similarity_scores) if similarity_scores else 0.0
    min_similarity_score = min(similarity_scores) if similarity_scores else 0.0

    # Calculate additional statistics
    similarity_std = np.std(similarity_scores) if similarity_scores else 0.0

    # Calculate percentage of high similarity matches (threshold-based)
    high_similarity_threshold = 0.8
    high_similarity_count = sum(1 for score in similarity_scores if score >= high_similarity_threshold)
    high_similarity_percentage = (high_similarity_count / total_samples * 100) if total_samples > 0 else 0.0

    # Calculate invalid prediction percentage
    invalid_percentage = (total_invalid_preds / total_samples * 100) if total_samples > 0 else 0.0

    result['exact_match_rate'] = exact_match_accuracy
    result['exact_match_rate_without_invalids'] = exact_match_accuracy_without_invalids
    result['avg_similarity_score'] = avg_similarity_score
    result['max_similarity_score'] = max_similarity_score
    result['min_similarity_score'] = min_similarity_score
    result['similarity_std'] = similarity_std
    result['high_similarity_percentage'] = high_similarity_percentage
    result['high_similarity_threshold'] = high_similarity_threshold
    result['total_samples'] = total_samples
    result['total_invalid_preds'] = total_invalid_preds
    result['invalid_percentage'] = invalid_percentage
    result['similarity_scores'] = similarity_scores

    return result

def find_data_files(dataset_dir: str) -> Dict[str, str]:
    test_dir = Path(dataset_dir)

    # Look for standard test files
    questions_file = test_dir / "v1_balanced_questions_test_scannetv2.json"
    annotations_file = test_dir / "v1_balanced_sqa_annotations_test_scannetv2.json"

    if not questions_file.exists():
        raise FileNotFoundError(f"Test questions file not found: {questions_file}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Test annotations file not found: {annotations_file}")

    # Images should be in scene subdirectories
    images_dir = test_dir
    return {
        "questions_file": str(questions_file),
        "annotations_file": str(annotations_file),
        "images_dir": str(images_dir)
    }

@dataclass
class DatasetResults:
    all_exact_matches: List[float] = field(default_factory=list)
    all_similarity_scores: List[float] = field(default_factory=list)
    normalized_preds: List[str] = field(default_factory=list)
    all_original_outputs: List[str] = field(default_factory=list)
    all_labels: List[str] = field(default_factory=list)
    total_batches: int = 0
    total_samples: int = 0
    eval_time: float = 0
    total_invalid_predictions: int = 0
    exact_match_rate: float = 0
    exact_match_rate_without_invalids: float = 0
    avg_similarity_score: float = 0
    max_similarity_score: float = 0
    min_similarity_score: float = 0
    similarity_std: float = 0
    high_similarity_percentage: float = 0
    high_similarity_threshold: float = 0.8
    invalid_percentage: float = 0
    similarity_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}

class MagmaSQA3DInference:

    def __init__(self, model_id: str = "microsoft/Magma-8B", device: str = None, dtype: str = "bf16"):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.model = None
        self.processor = None
        self.similarity_model = None
        logger.info(f"Initializing SQA3D inference with device: {self.device}, dtype: {dtype}")

    def fixed_magma_forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        **kwargs
    ):
        logger.debug(f"Input shapes: input_ids={input_ids.shape}, pixel_values={pixel_values.shape}")
        if pixel_values.dim() == 4 and pixel_values.shape[3] == 3:
            pixel_values = pixel_values.permute(0, 3, 1, 2)
        image_token_index = self.model.config.image_token_index
        for_inputs_embeds_ids = input_ids.clone()
        for_inputs_embeds_ids[input_ids == image_token_index] = 0
        inputs_embeds = self.model.get_input_embeddings()(for_inputs_embeds_ids)
        if pixel_values is not None and (input_ids == image_token_index).any():
            vision_output = self.model.vision_tower(pixel_values)
            image_features = vision_output['clip_vis_dense']
            b, c, h, w = image_features.shape
            image_features = image_features.flatten(2).transpose(1, 2)
            image_features = self.model.multi_modal_projector(image_features)
            new_inputs_embeds = []
            for batch_idx, cur_input_ids in enumerate(input_ids):
                image_token_idx_in_sequence = torch.where(cur_input_ids == image_token_index)[0]
                pre_image_embeds = inputs_embeds[batch_idx, :image_token_idx_in_sequence[0]]
                post_image_embeds = inputs_embeds[batch_idx, image_token_idx_in_sequence[0] + 1:]
                current_image_features = image_features[batch_idx]
                if current_image_features.dim() == 3:
                    current_image_features = current_image_features.squeeze(0)
                full_embeds = torch.cat([pre_image_embeds, current_image_features, post_image_embeds], dim=0)
                new_inputs_embeds.append(full_embeds)
            inputs_embeds = torch.stack(new_inputs_embeds, dim=0)
            new_sequence_length = inputs_embeds.shape[1]
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], new_sequence_length),
                dtype=torch.long,
                device=inputs_embeds.device
            )
            position_ids = torch.arange(
                0, new_sequence_length,
                dtype=torch.long,
                device=inputs_embeds.device
            ).unsqueeze(0)
        outputs = self.model.language_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        logits = self.model.language_model.lm_head(outputs[0]).float()
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error("NaN or Inf detected in logits")
        logger.debug(f"Logits shape: {logits.shape}")
        return PatchedMagmaCausalLMOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
    def load_model_and_processor(self):
        logger.info("Loading Magma model and processor...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.processor.tokenizer.padding_side = "left"
            self.model.forward = types.MethodType(self.fixed_magma_forward, self.model)
            logger.info("✓ Model and processor loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load model or processor: {e}")
            raise

    def load_similarity_model(self):
        """Load the similarity model for evaluation."""
        if self.similarity_model is None:
            logger.info("Loading similarity model for evaluation...")
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            logger.info("✓ Similarity model loaded successfully")
            

    def prepare_inputs(self, questions: Union[str, List[str]], images: Union[Image.Image, List[Image.Image], None] = None) -> Tuple[Dict[str, torch.Tensor], List[int]]:

        if isinstance(questions, str):
            questions = [questions]
        valid_indices = []
        valid_questions = []
        valid_images = []
        if images is None:
            logger.info(f"Skipping {len(questions)} samples due to missing images")
            return {}, []
        elif isinstance(images, Image.Image):
            images = [images]
        elif not isinstance(images, list):
            images = [images]
        for i, (question, img) in enumerate(zip(questions, images)):
            if img is not None:
                valid_indices.append(i)
                valid_questions.append(question)
                valid_images.append(img)
            else:
                logger.info(f"Skipping sample {i} due to missing image")
        if not valid_questions:
            logger.info("No valid samples with images found, skipping batch")
            return {}, []
        prompts = []
        for question in valid_questions:
            convs = [
                {"role": "system", "content": SQA3DDefinitions.SYSTEM_PROMPT},
                {"role": "user", "content": f"{question}\n<image>"}
            ]
            prompt = self.processor.tokenizer.apply_chat_template(
                convs, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
        inputs = self.processor(
            texts=prompts,
            images=valid_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)
            if torch.isnan(inputs['pixel_values']).any() or torch.isinf(inputs['pixel_values']).any():
                logger.error("NaN or Inf detected in pixel_values")
        return inputs, valid_indices

    def generate_response(self, inputs: Dict[str, torch.Tensor]) -> Union[str, List[str]]:

        try:
            with torch.inference_mode():
                generate_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    pixel_values=inputs.get('pixel_values'),
                    image_sizes=inputs.get('image_sizes'),
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            input_len = inputs['input_ids'].shape[1]
            batch_size = generate_ids.shape[0]
            generated_texts = []
            for i in range(batch_size):
                generated_tokens = generate_ids[i, input_len:]
                generated_text = self.processor.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                generated_texts.append(generated_text.strip())
            return generated_texts if batch_size > 1 else generated_texts[0]
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return [] if batch_size > 1 else ""

    def process_batch(self, questions: List[str], images: List[Image.Image] = None) -> Tuple[List[str], List[int]]:

        try:
            inputs, valid_indices = self.prepare_inputs(questions, images)
            if not valid_indices:
                return [], []
            responses = self.generate_response(inputs)
            if isinstance(responses, str):
                responses = [responses]
            return responses, valid_indices
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return [], []

    def evaluate_model(self, dataset_dir: str, batch_size: int = 8) -> Dict[str, Any]:

        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model_and_processor() first.")
        self.load_similarity_model()
        data_files = find_data_files(dataset_dir)
        logger.info("Found SQA3D data files:")
        logger.info(f"  Questions: {data_files['questions_file']}")
        logger.info(f"  Annotations: {data_files['annotations_file']}")
        logger.info(f"  Images: {data_files['images_dir']}")
        dataset_obj, dataloader = get_sqa3d_dataloader(
            questions_file=data_files['questions_file'],
            annotations_file=data_files['annotations_file'],
            images_dir=data_files['images_dir'],
            batch_size=batch_size
        )
        logger.info(f"Created dataloader with {len(dataset_obj)} samples")
        logger.info(f"Starting SQA3D evaluation with {len(dataloader)} batches...")
        dataset_results = DatasetResults()
        start_time = time.perf_counter()
        all_exact_matches = []
        all_similarity_scores = []
        all_normalized_preds = []
        all_original_outputs = []
        all_labels = []
        total_invalid_preds = 0
        for batch_idx, batch in enumerate(dataloader, 1):
            questions = batch['question']
            answers = batch['answer']
            scene_images = batch.get('scene_image', [None] * len(questions))
            logger.info(f"Processing batch {batch_idx}/{len(dataloader)} with {len(questions)} samples...")
            batch_outputs, valid_indices = self.process_batch(questions, scene_images)
            if valid_indices:
                valid_answers = [answers[i] for i in valid_indices]
                exact_matches, similarity_scores, invalid_preds, normalized_preds = validate_outputs_and_calculate_metrics(
                    self.similarity_model, batch_outputs, valid_answers
                )
                total_invalid_preds += invalid_preds
                all_exact_matches.extend(exact_matches)
                all_similarity_scores.extend(similarity_scores)
                all_normalized_preds.extend(normalized_preds)
                all_original_outputs.extend(batch_outputs)
                all_labels.extend(valid_answers)
                processed_samples = len(valid_indices)
            else:
                processed_samples = 0
            skipped_samples = len(questions) - processed_samples
            if skipped_samples > 0:
                logger.info(f"Skipped {skipped_samples} samples due to missing images")
            dataset_results.total_batches = batch_idx
            dataset_results.total_samples += processed_samples
            if batch_idx % 10 == 0:
                current_accuracy = sum(all_exact_matches) / len(all_exact_matches) if all_exact_matches else 0.0
                current_similarity = sum(all_similarity_scores) / len(all_similarity_scores) if all_similarity_scores else 0.0
                logger.info(f"Progress: {batch_idx} batches processed. Current accuracy: {current_accuracy:.4f}, Current avg similarity: {current_similarity:.4f}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        final_metrics = calculate_final_metrics(all_exact_matches, all_similarity_scores, total_invalid_preds)
        dataset_results.all_exact_matches = all_exact_matches
        dataset_results.all_similarity_scores = final_metrics["similarity_scores"]
        dataset_results.normalized_preds = all_normalized_preds
        dataset_results.all_original_outputs = all_original_outputs
        dataset_results.all_labels = all_labels
        dataset_results.exact_match_rate = final_metrics["exact_match_rate"]
        dataset_results.exact_match_rate_without_invalids = final_metrics["exact_match_rate_without_invalids"]
        dataset_results.avg_similarity_score = final_metrics["avg_similarity_score"]
        dataset_results.max_similarity_score = final_metrics["max_similarity_score"]
        dataset_results.min_similarity_score = final_metrics["min_similarity_score"]
        dataset_results.similarity_std = final_metrics["similarity_std"]
        dataset_results.high_similarity_percentage = final_metrics["high_similarity_percentage"]
        dataset_results.high_similarity_threshold = final_metrics["high_similarity_threshold"]
        dataset_results.total_invalid_predictions = final_metrics["total_invalid_preds"]
        dataset_results.invalid_percentage = final_metrics["invalid_percentage"]
        dataset_results.similarity_scores = final_metrics["similarity_scores"]
        end_time = time.perf_counter()
        dataset_results.eval_time = end_time - start_time
        logger.info("\nEvaluation completed!")
        logger.info(f"Total samples: {dataset_results.total_samples}")
        logger.info(f"Exact match rate: {dataset_results.exact_match_rate:.4f}")
        logger.info(f"Exact match rate (without invalids): {dataset_results.exact_match_rate_without_invalids:.4f}")
        logger.info(f"Average similarity score: {dataset_results.avg_similarity_score:.4f}")
        logger.info(f"High similarity percentage (≥{dataset_results.high_similarity_threshold}): {dataset_results.high_similarity_percentage:.2f}%")
        logger.info(f"Invalid predictions: {dataset_results.total_invalid_predictions} ({dataset_results.invalid_percentage:.2f}%)")
        logger.info(f"Evaluation time: {dataset_results.eval_time:.2f} seconds")
        return dataset_results.to_dict()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SQA3D inference with Magma model"
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Directory containing the SQA3D test dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./sqa3d_magma_inference_results',
        help='Directory to store inference results (default: ./sqa3d_magma_inference_results)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8)'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default='microsoft/Magma-8B',
        help='HuggingFace model identifier (default: microsoft/Magma-8B)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run inference on (cuda, cpu, etc.). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='bf16',
        choices=['fp16', 'bf16', 'fp32'],
        help='Model data type (default: bf16)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all samples)'
    )
    args = parser.parse_args()
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    return args

def main():
    """Main function to run SQA3D inference with Magma model."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Results will be stored in: {args.output_dir}")
    logger.info(f"Reading SQA3D dataset from: {args.dataset_dir}")
    try:
        sqa3d_inference = MagmaSQA3DInference(
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype
        )
        sqa3d_inference.load_model_and_processor()
        results = sqa3d_inference.evaluate_model(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size
        )
        results_file = os.path.join(args.output_dir, 'sqa3d_magma_inference_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"\nResults saved to: {results_file}")
        logger.info("\n=== SQA3D Magma Inference Results Summary ===")
        logger.info(f"Model: {args.model_id}")
        logger.info(f"Device: {sqa3d_inference.device}")
        logger.info(f"Total samples: {results.get('total_samples', 0)}")
        logger.info(f"Exact Match Rate: {results.get('exact_match_rate', 0):.4f}")
        logger.info(f"Exact Match Rate (without invalids): {results.get('exact_match_rate_without_invalids', 0):.4f}")
        logger.info(f"Average Similarity Score: {results.get('avg_similarity_score', 0):.4f}")
        logger.info(f"Max Similarity Score: {results.get('max_similarity_score', 0):.4f}")
        logger.info(f"Min Similarity Score: {results.get('min_similarity_score', 0):.4f}")
        logger.info(f"Similarity Std Dev: {results.get('similarity_std', 0):.4f}")
        logger.info(f"High Similarity (≥{results.get('high_similarity_threshold', 0.8)}): {results.get('high_similarity_percentage', 0):.2f}%")
        logger.info(f"Invalid predictions: {results.get('total_invalid_predictions', 0)} ({results.get('invalid_percentage', 0):.2f}%)")
        logger.info(f"Evaluation time: {results.get('eval_time', 0):.2f} seconds")
        logger.info("==============================================")
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())