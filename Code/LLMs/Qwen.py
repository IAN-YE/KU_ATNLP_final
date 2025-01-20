from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, Union, Optional
import json

from torch.nn import Module
from tqdm import tqdm

model_name = "../Models/Qwen2.5-1.5B-Instruct"
model_Parameter = "Qwen2.5-1.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

results = {}

def generate_text(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_text_batch(prompts: list, max_length: int = 100) -> list:
    inputs = tokenizer(prompts, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


base_prompt = """
### Instruction:
You are a task translator. Your job is to convert natural language instructions into sequences of action tokens.
- Output must only contain action tokens separated by spaces.
- Do not include any explanations or additional text in the output.
- Follow the format strictly: "I_TURN_RIGHT I_WALK ...".
    
### Examples:
{}

### New Input:
Input: {}
Output:
"""

def format_prompt(data, topk):
    src = data['src']
    tgt = data['tgt']
    similar_sentences = data['similar_sentences']
    similar_sentences_tgt = data['similar_sentences_tgt']

    example_data = []
    for i in range(topk):
        example_data.append((similar_sentences[15 - i], similar_sentences_tgt[15 - i]))

    examples = "\n".join(
        f'Input: "{example_input}"\nOutput: "{example_output}"'
        for example_input, example_output in example_data
    )

    prompt = base_prompt.format(examples, src)

    return prompt, tgt

import json
with open('top16_sentence.jsonl', 'r') as f:
    data = f.readlines()
    data = [json.loads(line) for line in data]


cnt = 1000

topk = 8

print(f"It's a test on model {model_Parameter}, with topk {topk}")

results = []
from tqdm import tqdm
for i in tqdm(data):
    prompt, tgt = format_prompt(i, topk)
    output = generate_text(prompt)
    results.append({
        "output": output,
        "tgt": tgt
    })

    cnt -= 1
    if cnt == 0:
        break

with open(f"results_{topk}_{model_Parameter}.json", "w") as f:
    json.dump(results, f)