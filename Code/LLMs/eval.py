import json
import re

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_new_input_block(text):
    text = text.replace("`", "")
    new_input_pattern = r"### New Input:\nInput: .+?\nOutput:\n(?:I_[A-Z_]+ ?)+"

    try:
        new_input_match = re.search(new_input_pattern, text, re.DOTALL).group()
    except AttributeError:
        return None, None
    input_pattern = r"Input:\s*(.*?)\n"
    output_pattern = r"Output:\s*(.*)"

    input_match = re.search(input_pattern, new_input_match, re.DOTALL)
    output_match = re.search(output_pattern, new_input_match, re.DOTALL)

    input_text = input_match.group(1).strip() if input_match else None
    output_text = output_match.group(1).strip() if output_match else None

    return input_text, output_text

if __name__ == "__main__":
    cnt = 0
    cnt_total = 0
    topkList = [2,4,8,16]
    model_size = [0.5, 1.5, 3, 7]
    data = read_json("results_8_Qwen2.5-7B-Instruct_exp2.json")
    for i in range(1000):
        text = data[i]['output']
        tgt = data[i]['tgt']
        input_res, output_res = extract_new_input_block(text)
        if output_res != None:
            cnt_total += 1
            if output_res == tgt:
                cnt += 1
    
    print(f"Topk: 8, Model Size: 7B, Accuracy: {cnt/cnt_total}, Total: {cnt_total}")
    
    # for t in topkList:
    #     for m in model_size:
    #         data = read_json(f"results_{t}_Qwen2.5-{m}B-Instruct.json")
    #         cnt = 0
    #         cnt_total = 0
    #         for i in range(len(data)):
    #             text = data[i]['output']
    #             tgt = data[i]['tgt']
    #             input_res, output_res = extract_new_input_block(text)
    #             if output_res != None:
    #                 cnt_total += 1
    #                 if output_res == tgt:
    #                     cnt += 1
    #         print(f"Topk: {t}, Model Size: {m}B, Accuracy: {cnt/cnt_total}, Total: {cnt_total}")
        

