from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import numpy as np
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
torch.cuda.empty_cache()

from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import login
from transformers.trainer_pt_utils import nested_detach
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from peft import PeftModel
# from lmflow.trainer.utils import compute_accuracy
#google-7b
from trl import DPOConfig, DPOTrainer
# from lmflow.trainer.dpo_trainer import DPOTrainer


#login('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')
#llama-7b
login('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')

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
)
from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

## Prepare training_args
pipeline_name = "finetuner"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))

if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

args = parser.parse_args()



tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, load_in_8bit=True)

if "llamassss" in model_args.model_name_or_path:
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
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)
print(pipeline_args.run_name)

if torch.distributed.get_rank() == 0:
    wandb.init(project='hh_ablation', name=pipeline_args.run_name)


#ref_model = AutoModelForCausalLM.from_pretrained(pipeline_args.loss, torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16, )
print(model_args.model_name_or_path)
print(model_args.use_lora)
if model_args.use_lora:
    model_lora = get_peft_model(model, peft_config)
    model_lora.print_trainable_parameters()
else:
    model_lora = model

def get_dataset(data_dir, split='train'):
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir)
    return dataset[split]

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]

# We also need to add a pad_token for the model. Otherwise, the reward model cannot handle a batch of inputs
model_lora.config.pad_token_id = tokenizer.eos_token_id
#assert model_lora.config.pad_token_id == tokenizer.pad_token_id

def build_dataset(tokenizer, config):
    ''' 
    We assume that we have preprocessed the dataset appropriately such that the sample is organized as follows:
    {"positive": prompt + answer_positive, "negative": prompt + answer_negative}, where the positive response is preferred.
    '''
    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt):],
            "rejected": sample["rejected"][len(prompt):],
        }
    
    data_dirs = ["harmless-base", "helpful-base", "helpful-rejection-sampled", "helpful-online"]

    # ds = concatenate_datasets([get_dataset(data_dir) for data_dir in data_dirs])
    ds = load_dataset("json", data_files=config.dataset_path)['train']
    ds = ds.map(split_prompt_and_responses)


    train_dataset = ds

    # ds = concatenate_datasets([get_dataset(data_dir, split='test') for data_dir in data_dirs])
    test_data_path = config.dataset_path.replace("train", "test")
    ds = load_dataset("json", data_files=test_data_path)['train']
    
    ds = ds.map(split_prompt_and_responses, batched=False)
    eval_dataset = ds

    return train_dataset, eval_dataset

#train_dataset, eval_dataset = build_dataset(tokenizer, data_args)
train_dataset, eval_dataset = build_dataset(tokenizer, data_args)

print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))

training_args = DPOConfig(
    output_dir=pipeline_args.output_dir,
    beta=0.1,
    learning_rate=pipeline_args.learning_rate,
    weight_decay=pipeline_args.weight_decay,
    lr_scheduler_type=pipeline_args.lr_scheduler_type,
    per_device_train_batch_size=pipeline_args.per_device_train_batch_size,
    per_device_eval_batch_size=pipeline_args.per_device_eval_batch_size,
    num_train_epochs=pipeline_args.num_train_epochs,
    bf16=pipeline_args.bf16,
    seed=pipeline_args.seed,
    do_train=pipeline_args.do_train,
    do_eval=pipeline_args.do_eval,
    run_name=pipeline_args.run_name,
    eval_strategy=pipeline_args.eval_strategy,
    eval_steps=pipeline_args.eval_steps,
    save_steps=pipeline_args.save_steps,
    deepspeed=pipeline_args.deepspeed,
)


trainer = DPOTrainer(
    model=model_lora,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=512,
    )

trainer.train()

if model_args.use_lora:
    model_lora = model_lora.merge_and_unload()
model_lora.save_pretrained(pipeline_args.output_dir, safe_serialization=False)
config = model.config
config.save_pretrained(pipeline_args.output_dir)
tokenizer.save_pretrained(pipeline_args.output_dir)