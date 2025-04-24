
import os
from enum import Enum

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)

from peft import LoraConfig


def create_datasets(tokenizer, data_args, training_args, apply_chat_template=True):
    def preprocess(samples):
        batch = []
        for conversation in samples["conversations"]:
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"content": batch}
    dataset = load_dataset(data_args.dataset_name, split='train')
    # #dataset = dataset.remove_columns_(dataset['id'])
    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    print(f"Size of the train set: {len(dataset)}")
    print(f"A sample of train dataset: {dataset[0]}")

    return dataset




def create_and_prepare_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        '''torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )'''
        model_config_torch_dtype = AutoConfig.from_pretrained(args.model_name_or_path).torch_dtype
        torch_dtype = (model_config_torch_dtype if model_config_torch_dtype in ["auto", None] else getattr(torch, str(model_config_torch_dtype).split(".")[1]))
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
        )

    peft_config = LoraConfig(
                            r=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    chatformat = ChatTemplate()
    tokenizer.chat_template = chatformat.chat_template
    tokenizer.pad_token = tokenizer.eos_token 


    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )
    

    return model, peft_config, tokenizer



class ChatTemplate:
    """Dataclass for special tokens used in ChatML, including system, user, and assistant tokens."""

    system_token: str = "<system>"
    user_token: str = "<user>"
    assistant_token: str = "<assistant>"
    eos_token: str = "</message>"

    @property
    def chat_template(self):
        return (
            "{% for message in messages %}"
            "{% if message['from'] == 'system' %}"
            f"{{{{'{self.system_token}' + message['value'] + '{self.eos_token}'}}}}"
            "{% elif message['from'] == 'human' %}"
            f"{{{{'{self.user_token}' + message['value'] + '{self.eos_token}'}}}}"
            "{% elif message['from'] == 'gpt' %}"
            f"{{{{'{self.assistant_token}' + message['value'] + '{self.eos_token}'}}}}"
            "{% endif %}"
            "{% endfor %}"
        )
