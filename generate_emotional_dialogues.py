from huggingface_hub import InferenceClient
import json
import torch
from transformers import pipeline
import random
import os

# Gán Hugging Face token
HF_TOKEN = ""

client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=HF_TOKEN,
)

# Tạo prompt dạng ChatML theo yêu cầu của LLaMA 3
messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]

# Xây dựng prompt thủ công (giống template của Hugging Face)
prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are a pirate chatbot who always responds in pirate speak!<|eot_id|>\n"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Who are you?<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

# Gọi API
response = client.text_generation(
    prompt=prompt,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    stop=["<|eot_id|>"],
)

print("Assistant:", response.strip())
