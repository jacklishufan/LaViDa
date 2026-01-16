
DREAM_7B_INST=/path/to/Dream-v0-Instruct-7B
VISION_MODEL_VERSION="/path/to/google/siglip-so400m-patch14-384"
LLM_VERSION=$DREAM_7B_INST
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
IMG_PATH_1M=/home/ubuntu/Open-Llava-V2
DATA_PATH=scripts/train/stage2.yaml
IMG_PATH=data/Open-LLaVA-Next
############### Pretrain ################
export ALWASY_DO_2DPOOL=1 

PROMPT_VERSION="dream"

MID_RUN_NAME="lavida-stage2-dream"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
NUM_GPUS=8
PORT=23334
export SELECT_ONE_INDEX=1
export DEBUG_FIX_PADDIN=1
BASE_RUN_NAME=/path/to/projectors
torchrun --nproc_per_node="${NUM_GPUS}"  --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path $DATA_PATH \
    --pretrain_mm_mlp_adapter="${BASE_RUN_NAME}/mm_projector.bin" \
    --image_folder $IMG_PATH \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "/outputdir/${MID_RUN_NAME}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --resume_from_checkpoint latest \
    --lmms_eval_generate_tasks=vqav2_val_lite,chartqa_lite,textvqa_val_lite,docvqa_val_lite,infovqa_val_lite \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}'
