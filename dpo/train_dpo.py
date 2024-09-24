'''
import os

import torch
from datasets import load_dataset, concatenate_datasets
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel

from macros import data_dirs

data_dirs = data_dirs[:2]
model_name_or_path = "RLHFlow/LLaMA3-SFT"
max_seq_length = 512
lora_r = 8
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_alpha = 32
lora_dropout = 0.1
lora_bias = "none"
beta = 0.1
learning_rate = 5e-5
weight_decay = 0.0001
lr_scheduler_type = 'constant'
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
num_train_epochs = 1
bf16 = True
seed = 14
do_train = True
do_eval = False
run_name = "dpo"
eval_strategy = "steps"
eval_steps = 1000

save_path = "outputs/dpo"
os.makedirs(save_path, exist_ok=True)


# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name_or_path,
    max_seq_length = max_seq_length,
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_r,
    target_modules = lora_target_modules,
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout, # Dropout = 0 is currently optimized
    bias = lora_bias,    # Bias = "none" is currently optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
)

training_args = DPOConfig(
    output_dir=save_path,
    beta=beta,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    lr_scheduler_type=lr_scheduler_type,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=num_train_epochs,
    bf16=bf16,
    seed=seed,
    do_train=do_train,
    do_eval=do_eval,
    run_name=run_name,
    eval_strategy=eval_strategy,
    eval_steps=eval_steps,
)
'''

import os
import sys

import torch
torch.cuda.empty_cache()

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from trl import DPOConfig, DPOTrainer


login('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')

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from macros import data_dirs

model_name = sys.argv[1]

models = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B"
}

# data_dirs = data_dirs[:2]
model_name_or_path = models[model_name]
max_seq_length = 512
use_lora = True
lora_r = 16
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_alpha = 32
lora_dropout = 0.1
lora_bias = "none"
beta = 0.1
learning_rate = 5e-5
weight_decay = 0.0001
lr_scheduler_type = 'constant'
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
num_train_epochs = 1
bf16 = True
seed = 14
do_train = True
do_eval = False
run_name = f"dpo_{model_name}"
eval_strategy = "steps"
eval_steps = 1000
save_steps = 10000
temp = 0.1

save_path = f"outputs/dpo_{model_name}"
os.makedirs(save_path, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, load_in_8bit=True)

if "llamassss" in model_name_or_path:
    tokenizer.add_special_tokens(
        {
            "eos_token": "[PAD]",
            "bos_token": "</s>",
            "unk_token": "</s>",
            "pad_token": "</s>",
        }
    )
else:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

## Get model, by default we use lora to accelerate training
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
if use_lora:
    model_lora = get_peft_model(model, peft_config)
    model_lora.print_trainable_parameters()
else:
    model_lora = model

def get_dataset(data_dir, split='train'):
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir)
    return dataset[split]

def separate_prompt(data_input):
    prompt, chosen = data_input['chosen'].rsplit("\n\nAssistant: ", 1)
    _, rejected = data_input['rejected'].rsplit("\n\nAssistant: ", 1)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

train_dataset = concatenate_datasets([get_dataset(data_dir) for data_dir in data_dirs])
train_dataset = train_dataset.map(separate_prompt)

test_dataset = concatenate_datasets([get_dataset(data_dir, split='test') for data_dir in data_dirs])
test_dataset = test_dataset.map(separate_prompt)

training_args = DPOConfig(
    output_dir=save_path,
    beta=beta,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    lr_scheduler_type=lr_scheduler_type,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    num_train_epochs=num_train_epochs,
    bf16=bf16,
    seed=seed,
    do_train=do_train,
    do_eval=do_eval,
    run_name=run_name,
    eval_strategy=eval_strategy,
    eval_steps=eval_steps,
    save_steps=save_steps,
)

dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_args,
    beta=temp,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    max_length=512,
)
dpo_trainer.train()

if use_lora:
    model_lora = model_lora.merge_and_unload()
model.save_pretrained(save_path, safe_serialization=False)
config = model.config
config.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)