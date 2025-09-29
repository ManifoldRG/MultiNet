#!/bin/bash

gpu_id=0

declare -a arr=(
    "microsoft/Magma-8B"
)

env_name=MoveNearGoogleBakedTexInScene-v0
# env_name=MoveNearGoogleBakedTexInScene-v1
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png

# URDF variations
declare -a urdf_version_arr=("recolor_cabinet_visual_matching_1" "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" None)

# Create a logs directory if it doesn't exist
mkdir -p logs

for ckpt_path in "${arr[@]}"; do 
    echo "Checkpoint path: $ckpt_path"
done

for urdf_version in "${urdf_version_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
        # Create a unique log file name
        timestamp=$(date +"%Y%m%d_%H%M%S")
        log_file="logs/Magma_${env_name}_${urdf_version}_${timestamp}.txt"
        
        echo "Starting experiment with URDF version: $urdf_version" >> "$log_file"
        echo "Checkpoint path: $ckpt_path" >> "$log_file"
        echo "Environment: $env_name" >> "$log_file"
        echo "Scene: $scene_name" >> "$log_file"
        echo "GPU ID: $gpu_id" >> "$log_file"
        echo "-------------------------------------------" >> "$log_file"

        CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_magma.py --policy-model magma \
            --ckpt-path ${ckpt_path} \
            --robot google_robot_static \
            --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
            --env-name ${env_name} --scene-name ${scene_name} \
            --rgb-overlay-path ${rgb_overlay_path} \
            --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
            --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
            --additional-env-build-kwargs urdf_version=${urdf_version} \
            --additional-env-save-tags baked_except_bpb_orange >> "$log_file" 2>&1 &

        echo "Job started on GPU $gpu_id"
	((gpu_id++))
    done
done

wait

echo "All experiments completed. Log files are in the 'logs' directory."
