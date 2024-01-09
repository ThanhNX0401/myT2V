import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_NAME = "vilsonrodrigues/falcon-7b-instruct-sharded"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


PEFT_MODEL = "ThanhNX/falcon_7b-FT"
config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(model, PEFT_MODEL)
def get_label_and_motion(prompts):
    generation_config = model.generation_config
    generation_config.max_new_tokens = 20
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    device = "cuda:0"

    prompt = f"""
    <human>: which direction is the object heading given the promt: {prompts}
    <assistant>:
    """.strip()

    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids = encoding.input_ids,
            attention_mask = encoding.attention_mask,
            generation_config = generation_config
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    lines = output.split("\n")
    assistant_line = next(line for line in lines if line.startswith("<assistant>:"))
    json_str = assistant_line[len("<assistant>: "):]

    # Parse the JSON string into a dictionary
    dictionary = json.loads(json_str)

    # Now 'dictionary' is a Python dictionary that you can work with
    print(dictionary)
    return dictionary['object'],dictionary['direction']
    
    