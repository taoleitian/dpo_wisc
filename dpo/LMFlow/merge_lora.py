from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

login('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')

model_id = "meta-llama/Meta-Llama-3-8B"
peft_model_id = "/nobackup2/samuelyeh/reliable_rf/outputs/sft_llama3_lora"
output_id = "/nobackup2/samuelyeh/reliable_rf/outputs/sft_llama3_lora_merge"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)

model.save_pretrained(output_id, safe_serialization=False)
config = model.config
config.save_pretrained(output_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(output_id)
