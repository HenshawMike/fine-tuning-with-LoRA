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