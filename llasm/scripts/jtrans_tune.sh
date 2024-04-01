deepspeed ../llasm/train/tune_jtrans.py \
    --deepspeed ./zero2.json \
    --model_name_or_path ../../models/vicuna-13b-v1.5 \
    --train_data_path ../../BinaryCorp/train_datasets \
    --eval_data_path ../../BinaryCorp/valid_datasets \
    --mm_encoder_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ../../models/llasm_finetune_jtrans \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --lora_enable True \
    --encoder ../encoder/jtrans/encoder \
    --tokenizer ../encoder/jtrans/tokenizer \
    --pretrain_mm_mlp_adapter ../../models/llasm_pretrain_jtrans/mm_projector.bin