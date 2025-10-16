eval_tasks=${1:-textvqa}
NUM_PROCESSES=${2:-4}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m accelerate.commands.launch --num_processes=$NUM_PROCESSES -m lmms_eval --model magma --model_args pretrained="microsoft/Magma-8B" \
    --tasks $eval_tasks --batch_size 1 --log_samples --log_samples_suffix magma_textvqa --output_path ./logs/
