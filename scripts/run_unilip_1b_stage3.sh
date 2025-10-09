#!/bin/bash
export OUTPUT_FOLDER=../work_dirs/1b_stage3
export EDIT_IMG_FOLDER=../data/edit_sft
export GEN_IMG_FOLDER=../data/gen_sft

# single node
# torchrun --nproc_per_node=4 --master_port=29506 
torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR \
    unilip/train/train_stage3.py \
    --deepspeed ../deepspeed_scripts/zero0.json \
    --model_name_or_path ../work_dirs/1b_stage2/checkpoint-xxx  \
    --unilip_path ../tokenizer_ckpt/1b_unilip.pth \
    --unilip_factor 10.6 \
    --mllm_path OpenGVLab/InternVL3-1B \
    --mllm_hf_path OpenGVLab/InternVL3-1B-hf \
    --vae_path mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers \
    --dit_path Efficient-Large-Model/Sana_600M_512px_diffusers \
    --version internvl \
    --data_type "mix" \
    --gen_image_folder ${GEN_IMG_FOLDER} \
    --edit_image_folder ${EDIT_IMG_FOLDER} \
    --gen_repeat 1 \
    --edit_repeat 3 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_FOLDER} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr":1e-5}' \
    --model_max_length 1024 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --n_query 256 \
    --n_und_query 0 \
    --report_to none \
    --run_name unilip_intern_vl_1b \
    --fix_dit False \
    --fix_connect False \
    --fix_llm True \
