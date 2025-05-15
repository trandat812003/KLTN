from huggingface_hub import InferenceClient
import json
import random
import os

# Gán Hugging Face token
HF_TOKEN = ""
client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)

# Instruction prefix
instruction = (
    "The following is a conversation between a Human and an empathetic AI assistant.\n"
    "The assistant is helpful, kind, and supportive. It responds to emotional concerns and offers support.\n\n"
)

# Một vài câu mở đầu
starting_posts = [
    "Human: I moved into a new state recently, and there’s a lot to do, but I don’t have any friends in the new place I stay at.\nAI:",
    "Human: I've been feeling really down lately, like nothing is going right.\nAI:",
    "Human: My dog passed away last week. He was like family to me.\nAI:",
    "Human: I failed my final project and feel like I’ve disappointed everyone.\nAI:",
    "Human: I just got laid off and don’t know how to tell my family.\nAI:",
]


# Hàm sinh một lượt phản hồi từ AI
def generate_turn(prompt, max_tokens=200):
    full_prompt = instruction + prompt
    response = client.text_generation(
        prompt=full_prompt,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        stop_sequences=["Human:", "AI:"],
    )
    return response.strip()


# Sinh đoạn hội thoại ~10 lượt
def generate_conversation(starting_prompt, turns=10):
    dialogue = starting_prompt.strip()
    for i in range(turns - 1):  # Đã có 1 lượt rồi
        if i % 2 == 0:
            # Human lượt kế tiếp
            dialogue += "\nHuman: " + generate_turn(dialogue + "\nHuman:")
        else:
            # AI phản hồi
            dialogue += "\nAI: " + generate_turn(dialogue + "\nAI:")
    return dialogue.strip()


# Tạo dữ liệu
augmented_data = []
for i in range(10):  # Tạo 10 đoạn hội thoại
    start = random.choice(starting_posts)
    full_dialogue = generate_conversation(start, turns=10)
    augmented_data.append(
        {
            "instruction": "Generate a full emotional support dialogue of ~10 turns.",
            "input": start.split("\n")[0].strip(),  # Lấy câu đầu tiên từ Human
            "output": full_dialogue,
        }
    )

# Lưu ra file JSONL
with open("augesc_llama3_api.jsonl", "w", encoding="utf-8") as f:
    for item in augmented_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ Generated multi-turn dialogues saved to augesc_llama3_api.jsonl")
