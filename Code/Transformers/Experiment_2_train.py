
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from Data_Processing.dataloader import MyDataset
from Data_Processing.vocab_manage import load_vocab
from Transformer.Transformer import Transformer

#Hyperparameters
# EMB_DIM = 128
# N_LAYERS = 1
# N_HEADS = 8
# FORWARD_DIM = 512
# DROPOUT = 0.05
# LEARNING_RATE = 7e-4
# BATCH_SIZE = 64
# GRAD_CLIP = 1
# Optimizer: AdamW

def train(model, dataloader, optimizer, criterion, grad_clip, device):
    print("Training!")
    model.train()
    epoch_loss = 0

    for i, batch in tqdm(list(enumerate(dataloader))):
        src = batch['src'].to(device)
        tgt = batch['tgt_output'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        optimizer.zero_grad()
        output = model(src, tgt_input)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt = tgt.contiguous().view(-1)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, device, top_k = 2):
    print(f"Evaluate!! Sampling from top {top_k}")
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0
    total_correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for i, batch in tqdm(list(enumerate(dataloader))):
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt = batch['tgt_output'].to(device)
            logits, generated_text = model.generate_top_k(
                src, tgt_input, max_len=tgt.shape[1],
                top_k = top_k)
            # print(logits.shape)
            # print(generated_text.shape)

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

    return token_acc, sequence_acc

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    path_train = '../Data/length_split/tasks_train_length.txt'
    path_test = '../Data/length_split/tasks_test_length.txt'
    vocab_in_path = '../Data/length_split/vocab_in.json'
    vocab_out_path = '../Data/length_split/vocab_out.json'
    # path_train = '../Data/simple_split/tasks_train_simple.txt'
    # path_test = '../Data/simple_split/tasks_test_simple.txt'
    # vocab_in_path = '../Data/simple_split/vocab_in.json'
    # vocab_out_path = '../Data/simple_split/vocab_out.json'

    vocal_in = load_vocab(vocab_in_path)
    vocal_out = load_vocab(vocab_out_path)

    Dataset_train = MyDataset(path_train, 64, vocal_in, vocal_out)
    Dataset_train.load_data()
    dataloader_train = DataLoader(Dataset_train, batch_size=64, shuffle=True)

    Dataset_test = MyDataset(path_test, 64, vocal_in, vocal_out)
    Dataset_test.load_data()
    dataloader_test = DataLoader(Dataset_test, batch_size=64, shuffle=False)
        
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
    # model.load_state_dict(torch.load('./model_2.pth'))
    # token_acc, seq_acc = evaluate(model, dataloader_test, device, top_k = 1)
    # print('token_acc:', token_acc, 'seq_acc:', seq_acc)
    
    # # print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4)
    criterion = nn.CrossEntropyLoss()

    # eval
    # subset = Subset(Dataset_test, range(6))
    # dataloader_subset = DataLoader(subset, batch_size=2, shuffle=True)

    epoch = 4
    for i in range(epoch):
        loss = train(model, dataloader_train, optimizer, criterion, 1, device)
        token_acc, seq_acc = evaluate(model, dataloader_test, device)
        print('epoch:', i, 'loss:', loss, 'token_acc:', token_acc, 'seq_acc:', seq_acc)

    # save model
    torch.save(model.state_dict(), 'model_2_1.pth')


if __name__ == "__main__":
    main()