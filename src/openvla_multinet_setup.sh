#!/bin/bash

# Check if the conda environment already exists
echo "Creating new conda environment 'multinet-openvla'..."
if ! conda info --envs | grep -q multinet-openvla; then
    conda create -n multinet-openvla python=3.10 -y
else
    echo "Conda environment 'multinet-openvla' already exists."
fi

# Activate the environment
source activate multinet-openvla
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# Multinet Dependencies
pip install -q -r requirements.txt

echo "Installing OpenVLA dependencies"
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

cd eval/profiling/openvla
pip install transformers draccus timm tensorflow==2.15.0 tensorflow-datasets==4.9.3 bitsandbytes
pip install -e .
pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

echo "Reminder: Please run the following command before executing openvla profiling script:"
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)/eval/profiling/openvla"
echo "conda activate multinet-openvla"

echo "python eval/profiling/openvla/experiments/robot/openvla_openx_profiling.py --openx_datasets_path ~/ManifoldRG/MultiNet/data/openx --dataset_statistics_path ~/ManifoldRG/MultiNet/src/eval/profiling/openvla/data/dataset_statistics.json --result_save_path ~/ManifoldRG/MultiNet/src"
