# Gemma-3-1B-IT LoRA Fine-Tuning (CPU-Only)

A lightweight, end-to-end fine-tuning project using **Google's
Gemma-3-1B-IT** model with **LoRA (Low-Rank Adaptation)** --- designed
to run **entirely on CPU**.

This repository demonstrates how to perform **parameter-efficient
fine-tuning (PEFT)** for specialized tasks (medical Q&A, code
generation, domain chatbots, etc.) **without a GPU**, making it ideal
for learning, experimentation, and portfolio projects on standard
laptops or desktops.

------------------------------------------------------------------------

## 🚀 Features

-   **Gemma-3-1B-IT** --- compact yet capable 1B-parameter
    instruction-tuned model (\~2 GB)
-   **LoRA fine-tuning** (CPU-only, no QLoRA)
-   **Tiny adapter size** after training (\~20--50 MB)
-   Built on the **Hugging Face ecosystem**
-   **Local CPU inference**
-   Ready for Hugging Face Hub, Spaces, and Inference Endpoints

------------------------------------------------------------------------

## 📋 Prerequisites

-   Python 3.10+
-   16 GB RAM recommended (8 GB may work with small batches)
-   CPU-only machine (GPU not required)

------------------------------------------------------------------------

## 📦 Installation

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets peft trl accelerate huggingface_hub
```

------------------------------------------------------------------------

## 📥 Model Download

Accept Google's license on the model page before downloading.

``` bash
huggingface-cli login
huggingface-cli download google/gemma-3-1b-it --local-dir ./google/gemma-3-1b-it
```

Model size: \~2.04 GB

------------------------------------------------------------------------

## 🛠 Fine-Tuning (LoRA on CPU)

Main script:

``` text
scripts/finetune_lora.py
```

### Example Dataset

``` python
from datasets import load_dataset
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
```

Output adapter is saved to:

``` text
./output/gemma-lora-medical/
```

------------------------------------------------------------------------

## ▶️ Local Inference

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    "./google/gemma-3-1b-it",
    device_map="cpu"
)

model = PeftModel.from_pretrained(model, "./output/gemma-lora-medical")
tokenizer = AutoTokenizer.from_pretrained("./google/gemma-3-1b-it")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("What are the symptoms of hypertension?")[0]["generated_text"])
```

Expected speed: \~5--15 tokens/sec on modern CPUs.

------------------------------------------------------------------------

## 🌐 Deployment

-   Push to Hugging Face Hub
-   Create a free Gradio Space
-   Deploy via Hugging Face Inference Endpoints

------------------------------------------------------------------------

## 📁 Project Structure

``` text
.
├── google/gemma-3-1b-it/
├── scripts/finetune_lora.py
├── output/gemma-lora-medical/
├── inference_test.py
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## 💡 Why This Project Rocks

-   Demonstrates modern PEFT techniques
-   Runs fully on CPU
-   End-to-end ML workflow
-   Perfect for portfolios and demos

------------------------------------------------------------------------

## 📄 License

-   Base model: Google Gemma License
-   Code: MIT License

Built with ❤️ on CPU-only hardware