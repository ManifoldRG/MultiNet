#!/bin/bash

ckpt_path=${1:-"microsoft/Magma-8B"}
policy_model=${2:-"magma"}
action_ensemble_temp=${3:-"0"}
logging_dir=${4:-"logs/drawer_matching_eval"}
gpu_id=0  

mkdir -p ${logging_dir}

declare -a env_names=(
    PlaceIntoClosedTopDrawerCustomInScene-v0
    # PlaceIntoClosedMiddleDrawerCustomInScene-v0
    # PlaceIntoClosedBottomDrawerCustomInScene-v0
)

declare -a urdf_version_arr=("recolor_cabinet_visual_matching_1" "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "None")

run_task() {
    local gpu=$1
    local env_name=$2
    local overlay_path=$3
    local robot_x=$4
    local robot_y=$5
    local robot_rpy_z=$6
    local urdf_version=$7
    local tag=$8

    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="${logging_dir}/eval_${env_name}_${urdf_version}_${tag}_${timestamp}.log"
    
    echo "Starting task on GPU ${gpu}: ${env_name} ${tag} (urdf: ${urdf_version})"
    
    CUDA_VISIBLE_DEVICES=${gpu} python simpler_env/main_inference_magma.py \
        --policy-model ${policy_model} \
        --ckpt-path ${ckpt_path} \
        --logging-dir ${logging_dir} \
        --robot google_robot_static \
        --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
        --env-name ${env_name} --scene-name dummy_drawer \
        --robot-init-x ${robot_x} ${robot_x} 1 --robot-init-y ${robot_y} ${robot_y} 1 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 ${robot_rpy_z} ${robot_rpy_z} 1 \
        --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
        --rgb-overlay-path ${overlay_path} \
        --enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=${urdf_version} model_ids=baked_apple_v2 > ${log_file} 2>&1 &
    
    echo "Job started on GPU ${gpu}"
}

process_task() {
    local env_name=$1
    local urdf_version=$2
    
    # A0
    run_task ${gpu_id} ${env_name} "./SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png" "0.644" "-0.179" "-0.03" ${urdf_version} "A0"
    
    ((gpu_id++))
    if [ $gpu_id -eq 4 ]; then
        echo "waiting for current batch tasks to complete..."
        wait
        echo "current batch tasks completed"
        gpu_id=0
    fi
    
    # B0
    run_task ${gpu_id} ${env_name} "./SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png" "0.652" "0.009" "0" ${urdf_version} "B0"
    
    ((gpu_id++))
    if [ $gpu_id -eq 4 ]; then
        echo "waiting for current batch tasks to complete..."
        wait
        echo "current batch tasks completed"
        gpu_id=0
    fi
    
    # C0
    run_task ${gpu_id} ${env_name} "./SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png" "0.665" "0.224" "0" ${urdf_version} "C0"
    
    ((gpu_id++))
    if [ $gpu_id -eq 4 ]; then
        echo "waiting for current batch tasks to complete..."
        wait
        echo "current batch tasks completed"
        gpu_id=0
    fi
}

echo "Start evaluation - total ${#urdf_version_arr[@]} URDF variants × ${#env_names[@]} environments × 3 positions = $((${#urdf_version_arr[@]} * ${#env_names[@]} * 3)) tasks"

for urdf_version in "${urdf_version_arr[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "processing URDF=${urdf_version}, ENV=${env_name}"
        process_task "${env_name}" "${urdf_version}"
    done
done

if [ $gpu_id -ne 0 ]; then
    echo "waiting for last batch tasks to complete..."
    wait
    echo "last batch tasks completed"
fi

echo "All evaluations completed! Results saved in ${logging_dir}"
