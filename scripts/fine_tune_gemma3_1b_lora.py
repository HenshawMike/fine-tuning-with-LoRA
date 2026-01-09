import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftConfig
from trl import  SFTTrainer, SFTConfig

BASE_MODEL_PATH = ".google/gemma-3-1b-it"
OUTPUT_DIR = "./output/gemma-lora-medical"


#loading tokenizer  and model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype= torch.float32,
    device_map= "cpu",
    low_cpu_mem_usage = True,
    )

peft_config=LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ["q_proj", "k_proj", "v_proj", "o_proj"],
    )

model = get_peft_model(model, peft_config)

dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
dataset = dataset["train"].train_test_split(test_size=0.1)

def format_example(example):
    messages =[
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    ]
    return {"text": tokenize.apply_chat_template(messages, 
            tokenize=False, 
            add_generation_prompt=False)}

dataset=dataset.map(format_example)