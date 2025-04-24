CUDA_VISIBLE_DEVICES=3 python train.py \
    --seed 42 \
    --model_name_or_path "mistralai/Mistral-7B-v0.3" \
    --dataset_name "castorini/rank_zephyr_training_data" \
    --add_special_tokens False \
    --append_concat_token False \
    --splits "train,val,test" \
    --max_seq_len 32768 \
    --num_train_epochs 5 \
    --logging_steps 5 \
    --log_level "info" \
    --logging_strategy "steps" \
    --evaluation_strategy "no" \
    --do_eval False \
    --save_strategy "epoch" \
    --push_to_hub \
    --hub_private_repo True \
    --hub_strategy "every_save" \
    --bf16 True \
    --packing False \
    --optim adamw_bnb_8bit \
    --learning_rate 5e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 200 \
    --output_dir "outputs/Mistral" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --use_reentrant True \
    --dataset_text_field "content" 
    # --use_peft_lora False \
    # --use_flash_attn True \
    # --use_peft_lora True \
    # --lora_r 8 \
    # --lora_alpha 16 \
    # --lora_dropout 0.1 \
    # --lora_target_modules "all-linear" #\
    # --report_to "wandb" \
    # --run_name "test1" #\

# --chat_template_format "chatml" \
# --weight_decay 1e-4 \
# --warmup_ratio 0.0 \
# --max_grad_norm 1.0 \
# --use_peft_lora True \
# --lora_r 8 \
# --lora_alpha 16 \
# --lora_dropout 0.1 \
# --lora_target_modules "all-linear" \
# --use_4bit_quantization True \
# --use_nested_quant True \
# --bnb_4bit_compute_dtype "bfloat16" \
# --packing False \
# --wandb_project "test1"