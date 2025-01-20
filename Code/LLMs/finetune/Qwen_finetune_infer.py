from unsloth import FastLanguageModel

max_seq_length = 1024
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False 

model_path = "finetune_ATNLP/QWEN2.5-7B/checkpoint-4000" # YOUR MODEL YOU USED FOR TRAINING
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "finetune_ATNLP/QWEN2.5-7B/checkpoint-4000", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)




import json
from tqdm import tqdm

results = []

with open('Prompt_test/top16_sentence_exp2.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

cnt = 0

topk = 8

for case in tqdm(data):
    cnt += 1
    if cnt < 1000:
        continue
    src = case['src']
    tgt = case['tgt']
    similar_sentences = case['similar_sentences']
    similar_sentences_tgt = case['similar_sentences_tgt']

    example_data = []
    for i in range(topk):
        example_data.append((similar_sentences[15 - i], similar_sentences_tgt[15 - i]))

    examples = "\n".join(
        f'Input: "{example_input}"\nOutput: "{example_output}"'
        for example_input, example_output in example_data
    )

    inputs = tokenizer(
            [
                alpaca_prompt.format(
                    "",
                    f"{src}",
                    "",
                )
            ], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 1024, use_cache = True)
    output_text = tokenizer.batch_decode(outputs)[0]
    response_start = output_text.find("### Response:") + len("### Response:")
    response_end = output_text.find("<|endoftext|>")
    response = output_text[response_start:response_end].strip()

    results.append({
        "output": response,
        "tgt": tgt
    })



with open(f"results_finetune_zero_2.json", "w") as f:
    json.dump(results, f)