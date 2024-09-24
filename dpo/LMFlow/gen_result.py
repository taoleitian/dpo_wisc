import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, Callable, Tuple

sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

import deepspeed 
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
torch.cuda.empty_cache()

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch import LongTensor, FloatTensor
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.trainer_pt_utils import nested_detach
# from lmflow.trainer.utils import compute_accuracy
#google-7b
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import pad_to_length
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

# if torch.distributed.get_rank() == 0:
#     wandb.init(project='eval', name=pipeline_args.run_name)


ref_model = AutoModelForCausalLM.from_pretrained('/nobackup2/samuelyeh/reliable_rf/outputs/sft_llama3', torch_dtype=torch.bfloat16, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")


def get_dataset(data_dir, split='train'):
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir)
    return dataset[split]

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]

def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt):],
            "rejected": sample["rejected"][len(prompt):],
        }

# We also need to add a pad_token for the model. Otherwise, the reward model cannot handle a batch of inputs
model.config.pad_token_id = tokenizer.eos_token_id
#assert model_lora.config.pad_token_id == tokenizer.pad_token_id

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

def load_eval_set(eval):
    ds = load_dataset("json", data_files=f"/u/s/a/samuelyeh/private/reliable_human_feedback/datasets/{eval}/test.jsonl")['train']
    ds = ds.map(split_prompt_and_responses, batched=False)
    return ds

os.makedirs('reliable_hf/gen_result', exist_ok=True)

filename = model_args.model_name_or_path.split('/')[-1]

result = []

eval_dataset = load_eval_set('normal')
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    beta=0.1,
    train_dataset=eval_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=512,
    )

def create_stopping_criteria(stop_words, tokenizer):

    class StoppingCriteriaSub(StoppingCriteria):
        def __init__(self, stops = [], encounters = 1):
            super().__init__()
            self.stops = stops

        def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> bool:
            last_token = input_ids[0][-1]
            for stop in self.stops:
                if tokenizer.decode(stop) == tokenizer.decode(last_token):
                    return True
            return False

    stop_word_ids = [tokenizer(stop_word,
                               return_tensors="pt", 
                               add_special_tokens=False)["input_ids"].squeeze() 
                               for stop_word in stop_words]

    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_word_ids)])
    return stopping_criteria

stop_words_list = ["Human:", '\n\n']
stopping_criteria = create_stopping_criteria(stop_words_list, tokenizer)

dataloader = trainer.get_eval_dataloader()
for d in tqdm.tqdm(dataloader):
    batch = trainer._prepare_inputs(d)
    policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=512,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
    
    policy_output = pad_to_length(policy_output, 512, tokenizer.pad_token_id)
    policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)
    
    for i, (prompt, pol) in enumerate(zip(batch["prompt"], policy_output_decoded)):
        search_term_idx = [pol[len(prompt):].find(search_term) for search_term in stop_words_list]
        search_term_idx = min([idx if idx != -1 else len(pol) for idx in search_term_idx])
        if pol[len(prompt):len(prompt)+search_term_idx] == "":
            print(f"@@@ {i}", pol[len(prompt):])
        result.append({
            "prompt": prompt,
            "output": pol[len(prompt):len(prompt)+search_term_idx],
        })

json.dump(result, open(f'reliable_hf/gen_result/{filename}.json', 'w'))