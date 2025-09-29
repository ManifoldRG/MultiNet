#!/bin/bash

policy_model="magma"
ckpt_path="microsoft/Magma-8B"
logging_dir=${1:-"logs/drawer_variant_eval"}
gpu_id=0  

mkdir -p ${logging_dir}

declare -a env_names=(
    PlaceIntoClosedTopDrawerCustomInScene-v0
)

start_task() {
    local env_name=$1
    local scene_name=$2
    local task_name=$3
    local extra_args=$4
    
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="${logging_dir}/eval_${env_name}_${scene_name}_${task_name}_${timestamp}.log"
    
    echo "Starting task on GPU ${gpu_id}: ${env_name} ${scene_name} (${task_name})"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_magma.py \
        --policy-model ${policy_model} \
        --ckpt-path ${ckpt_path} \
        --logging-dir ${logging_dir} \
        --robot google_robot_static \
        --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.65 0.65 1 --robot-init-y -0.2 0.2 3 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
        --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
        ${extra_args} >> ${log_file} 2>&1 &
    
    echo "Job started on GPU ${gpu_id}"
    
    ((gpu_id++))
    if [ $gpu_id -eq 4 ]; then
        echo "waiting for current batch tasks to complete..."
        wait
        echo "current batch tasks completed"
        gpu_id=0
    fi
}

scene_name=frl_apartment_stage_simple
for env_name in "${env_names[@]}"; do
    start_task "$env_name" "$scene_name" "base" "--enable-raytracing --additional-env-build-kwargs model_ids=apple"
done

declare -a scene_names=(
    "modern_bedroom_no_roof"
    "modern_office_no_roof"
)
for scene_name in "${scene_names[@]}"; do
    for env_name in "${env_names[@]}"; do
        start_task "$env_name" "$scene_name" "bg" "--additional-env-build-kwargs shader_dir=rt model_ids=apple"
    done
done

scene_name=frl_apartment_stage_simple
for env_name in "${env_names[@]}"; do
    start_task "$env_name" "$scene_name" "brighter" "--additional-env-build-kwargs shader_dir=rt light_mode=brighter model_ids=apple"
    start_task "$env_name" "$scene_name" "darker" "--additional-env-build-kwargs shader_dir=rt light_mode=darker model_ids=apple"
done

scene_name=frl_apartment_stage_simple
for env_name in "${env_names[@]}"; do
    start_task "$env_name" "$scene_name" "station2" "--additional-env-build-kwargs shader_dir=rt station_name=mk_station2 model_ids=apple"
    start_task "$env_name" "$scene_name" "station3" "--additional-env-build-kwargs shader_dir=rt station_name=mk_station3 model_ids=apple"
done

if [ $gpu_id -ne 0 ]; then
    echo "waiting for last batch tasks to complete..."
    wait
    echo "last batch tasks completed"
fi

echo "All evaluations completed! Results saved in ${logging_dir}"
