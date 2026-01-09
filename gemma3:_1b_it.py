from transformers import pipeline

pipe= pipeline(
    "text-generation",
    model="/home/henshawmikel/google/gemma-3-1b-it",
    device_map="cpu",
    dtype="auto",
    max_new_tokens=512,
)

prompt = "Explain benefits of sleeping."
result = pipe(prompt)
print(result[0]["generated_text"])