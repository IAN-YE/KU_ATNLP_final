import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.nn.functional as F

from Data_Processing.dataloader import MyDataset
from Data_Processing.vocab_manage import load_vocab
from Transformer.Transformer import Transformer


def plot_acc(action_sequence_length, accuracy, xlabel, ylabel, title, file_name):
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 6))  # Set figure size

    # Plot bar chart
    ax1.bar(action_sequence_length, accuracy, width=1.0, edgecolor='black')

    # Customize axes and labels
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color="black")
    ax1.set_title(title)

    # Add gridlines for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot as an image file
    plt.savefig(file_name, dpi=300, bbox_inches='tight')

    # Display the plot
    # plt.show()

def evaluate(model, dataloader, device, top_k = 2, oracle=True):
    print(f"Sampling from top {top_k}")
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0
    total_correct_sequences = 0
    total_sequences = 0

    with torch.no_grad(): # batch size should be 1 for this experiment
        for i, batch in tqdm(list(enumerate(dataloader))):
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt = batch['tgt_output'].to(device)
            # find the first 0 index in tgt[0]
            
            if oracle:
                first_zero_index = torch.argmax((tgt == 0).int())
                logits, generated_text = model.generate_with_oracle(
                    src, tgt, max_len=first_zero_index.item(),
                    top_k = top_k, oracle=True)
            else:
                logits, generated_text = model.generate_top_k(
                    src, tgt, max_len=tgt.shape[1],
                    top_k = top_k)
            # print(logits.shape)
            print('generated_text:', generated_text)
            print('tgt:', tgt)

            target_length = tgt.shape[1]

            current_length = generated_text.size(1)
            padding_length = target_length - current_length

            if padding_length > 0:
                generated_text = F.pad(generated_text, (0, padding_length), mode='constant', value=0)
            else:
                generated_text = generated_text

            print(generated_text)

            # Token-level accuracy
            correct_tokens = (generated_text == tgt)
            non_zero_mask = (tgt != 0) # Evaluation time Oracle 
            masked_correct_tokens = correct_tokens & non_zero_mask
            total_correct_tokens += masked_correct_tokens.sum().item()
            total_tokens += non_zero_mask.sum().item()

            # Sequence-level accuracy
            generated_text[tgt == 0] = 0 # Evaluation time Oracle 
            correct_sequences = (generated_text == tgt).all(dim=1)
            # print(correct_sequences)
            total_correct_sequences += correct_sequences.sum().item()
            total_sequences += tgt.size(0)

            # print(tgt, pred_sequences)

    token_acc = total_correct_tokens / total_tokens
    sequence_acc = total_correct_sequences / total_sequences

    return token_acc, sequence_acc, total_tokens, total_sequences, total_correct_tokens, total_correct_sequences

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    path_test = '../Data/length_split/tasks_test_length.txt'
    vocab_in_path = '../Data/length_split/vocab_in.json'
    vocab_out_path = '../Data/length_split/vocab_out.json'
    # path_test = '../Data/simple_split/tasks_test_simple.txt'
    # vocab_in_path = '../Data/simple_split/vocab_in.json'
    # vocab_out_path = '../Data/simple_split/vocab_out.json'

    vocal_in = load_vocab(vocab_in_path)
    vocal_out = load_vocab(vocab_out_path)

    token_acc_command_statistics = defaultdict(list)
    token_acc_action_statistics = defaultdict(list)
    seq_acc_command_statistics = defaultdict(list)
    seq_acc_action_statistics = defaultdict(list)
    # Initialize containers
    command_sequence_length = [4, 6, 7, 8, 9]
    for item in command_sequence_length:
        token_acc_command_statistics[item]
        seq_acc_command_statistics[item]
    action_sequence_length = [32, 33, 36, 40, 48, 24, 25, 26, 27, 28, 30]
    for item in action_sequence_length:
        token_acc_action_statistics[item]
        seq_acc_action_statistics[item]

    model = Transformer(
        emb_dim=128,
        num_layers=2,
        num_heads=8,
        forward_dim=256,
        dropout=0.15,
        src_vocab_size=len(vocal_in),
        tgt_vocab_size=len(vocal_out),
        src_pad_idx=0,
        tgt_pad_idx=0,
    ).to(device)
    model.load_state_dict(torch.load('./model_2_1.pth'))

    command_total_tokens = 0
    command_total_sequences = 0
    command_total_correct_tokens = 0
    command_total_correct_sequences = 0

    action_total_tokens = 0
    action_total_sequences = 0
    action_total_correct_tokens = 0
    action_total_correct_sequences = 0


    for key in token_acc_command_statistics:
        if key != 6:
            continue
        Dataset_test = MyDataset(path_test, 1, vocal_in, vocal_out)
        Dataset_test.load_data(command_length=key)
        print("Command length:", key, " Dataset size:", Dataset_test.__len__())
        dataloader_test = DataLoader(Dataset_test, batch_size=1, shuffle=False)
        token_acc, seq_acc, total_tokens, total_sequences, total_correct_tokens, total_correct_sequences  = evaluate(model, dataloader_test, device, top_k = 1, oracle=True)
        print('Command length:', key, 'token_acc:', token_acc, 'seq_acc:', seq_acc)
        token_acc_command_statistics[key].append(token_acc)
        seq_acc_command_statistics[key].append(seq_acc)
        command_total_tokens += total_tokens
        command_total_sequences += total_sequences
        command_total_correct_tokens += total_correct_tokens
        command_total_correct_sequences += total_correct_sequences
        

    # for key in token_acc_action_statistics:
    #     Dataset_test = MyDataset(path_test, 1, vocal_in, vocal_out)
    #     Dataset_test.load_data(action_length=key)
    #     print("Action length:", key, " Dataset size:", Dataset_test.__len__())
    #     dataloader_test = DataLoader(Dataset_test, batch_size=1, shuffle=False)
    #     token_acc, seq_acc, total_tokens, total_sequences, total_correct_tokens, total_correct_sequences = evaluate(model, dataloader_test, device, top_k = 1, oracle=True)
    #     print('Action length:', key, 'token_acc:', token_acc, 'seq_acc:', seq_acc)
    #     token_acc_action_statistics[key].append(token_acc)
    #     seq_acc_action_statistics[key].append(seq_acc)
    #     action_total_tokens += total_tokens
    #     action_total_sequences += total_sequences
    #     action_total_correct_tokens += total_correct_tokens
    #     action_total_correct_sequences += total_correct_sequences

    acc_dict = {
        'token_acc_command_statistics': token_acc_command_statistics,
        'seq_acc_command_statistics': seq_acc_command_statistics,
        'token_acc_action_statistics': token_acc_action_statistics,
        'seq_acc_action_statistics': seq_acc_action_statistics
    }

    # with open('acc_dict_without_oracle.json', 'w') as f:
    #     json.dump(acc_dict, f, indent=4)
    
    # accuracy_token_acc_action = [token_acc_action_statistics[len][0] for len in action_sequence_length]
    # plot_acc(action_sequence_length, accuracy_token_acc_action, 
    #     "Ground-Truth Action Sequence Length (in words)", 
    #     '"Accuracy on New Commands (%)"', 
    #     "Token-Level Accuracy by Action Sequence Length", 
    #     'accuracy_token_acc_action.png')
    
    # accuracy_seq_acc_action = [seq_acc_action_statistics[len][0] for len in action_sequence_length]
    # plot_acc(action_sequence_length, accuracy_seq_acc_action, 
    #     "Ground-Truth Action Sequence Length (in words)", 
    #     '"Accuracy on New Commands (%)"', 
    #     "Sequence-Level Accuracy by Action Sequence Length", 
    #     'accuracy_seq_acc_action.png')

    # accuracy_token_acc_command = [token_acc_command_statistics[len][0] for len in command_sequence_length]
    # plot_acc(command_sequence_length, accuracy_token_acc_command, 
    #     "Command Length (in words)", 
    #     '"Accuracy on New Commands (%)"', 
    #     "Token-Level Accuracy by Action Sequence Length", 
    #     'accuracy_token_acc_command.png')

    # accuracy_seq_acc_command = [seq_acc_command_statistics[len][0] for len in command_sequence_length]
    # plot_acc(command_sequence_length, accuracy_seq_acc_command, 
    #     "Command Length (in words)", 
    #     '"Accuracy on New Commands (%)"', 
    #     "Sequence-Level Accuracy by Action Sequence Length", 
    #     'accuracy_seq_acc_command.png')

if __name__ == "__main__":
    main()