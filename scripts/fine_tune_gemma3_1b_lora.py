import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, training_args
from peft import LoraConfig, get_peft_model, PeftConfig
from trl import  SFTTrainer, SFTConfig

BASE_MODEL_PATH = "google/gemma-3-1b-it"
OUTPUT_DIR = "output/gemma-lora-medical"


#loading tokenizer  and model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype= torch.float32,
    device_map= "cpu",
    low_cpu_mem_usage = True,
    )
#loading peft config for lora
peft_config=LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ["q_proj", "k_proj", "v_proj", "o_proj"],
    )

model = get_peft_model(model, peft_config)

# Load and inspect the dataset
print("Loading dataset...")
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")

# Print dataset structure for debugging
print("\nDataset structure:", dataset)
print("\nFirst example in train set:", dataset["train"][0])

# Split the dataset
dataset = dataset["train"].train_test_split(test_size=0.1)

def format_example(example):
    # Debug: Print example keys
    if 'instruction' not in example and 'input' in example:
        # Try to handle different dataset formats
        user_prompt = example.get("input", "")
        assistant_prompt = example.get("output", "")
    else:
        # Original format
        user_prompt = example.get("instruction", example.get("question", ""))
        assistant_prompt = example.get("output", example.get("answer", ""))
    
    # Skip empty examples
    if not user_prompt and not assistant_prompt:
        print("Warning: Empty example:", example)
        return None
    
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt}
    ]
    
    try:
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": formatted_text}
    except Exception as e:
        print(f"Error formatting example: {e}")
        print("Problematic example:", example)
        return None

# Apply formatting and filter out any None values
print("\nFormatting dataset...")
dataset = dataset.map(format_example)
print(f"\nDataset after formatting - Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

training_args= SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate= 2e-4,
    fp16= False,
    logging_steps= 10,
    save_strategy ="epoch",
    report_to ="none",
    bf16=False,        # Disable bf16
    tf32=False,
    max_seq_length=512,
    dataset_text_field="text",
    packing=False
   #push_to_hub=True,           
   #hub_model_id="your-username/gemma-3-medical-lora",
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    args=training_args,
           # Optional: forces CPU
                  # Avoids triggering AutoProcessor
)
    
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR}")