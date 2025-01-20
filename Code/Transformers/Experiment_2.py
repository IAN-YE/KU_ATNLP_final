import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Data_Processing.dataloader import MyDataset
from Data_Processing.vocab_manage import load_vocab
from Transformer.Transformer import Transformer

from Experiment_1a import train, evaluate

# #Hyperparameters
# # EMB_DIM = 128 
# # N_LAYERS = 2 
# # N_HEADS = 8 
# # FORWARD_DIM = 256 
# # DROPOUT = 0.15 
# # LEARNING_RATE = 2e-4 
# # GRAD_CLIP = 1 
# # BATCH_SIZE = 16 
# # Optimizer: AdamW

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('device:', device)
# path = '../Data/length_split/tasks_train_length.txt'
# Dataset_train = MyDataset(path, 16)
# Dataset_train.load_data()
# dataloader_train = DataLoader(Dataset_train, batch_size=16, collate_fn=lambda batch: collate_fn(batch, max_len=64), shuffle=True)

# model = Transformer(
#     emb_dim=128,
#     num_layers=2,
#     num_heads=8,
#     forward_dim=256,
#     dropout=0.15,
#     src_vocab_size=len(Dataset_train.vocab_in),
#     tgt_vocab_size=len(Dataset_train.vocab_out),
#     src_pad_idx=0,
#     tgt_pad_idx=0,
# ).to(device)

# # print(model)
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
# criterion = nn.CrossEntropyLoss(ignore_index=0)

# # eval
# path = '../Data/length_split/tasks_test_length.txt'
# Dataset_test = MyDataset(path, 16)
# Dataset_test.load_data()
# dataloader_test = DataLoader(Dataset_test, batch_size=16, collate_fn=lambda batch: collate_fn(batch, max_len=64), shuffle=False)

# epoch = 5
# for i in range(epoch):
#     loss = train(model, dataloader_train, optimizer, criterion, 1, device)
#     token_acc, seq_acc = evaluate(model, dataloader_test, device)
#     print('epoch:', i, 'loss:', loss, 'token_acc:', token_acc, 'seq_acc:', seq_acc)



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from Data_Processing.dataloader import MyDataset
from Transformer.Transformer import Transformer

#Hyperparameters
# EMB_DIM = 128
# N_LAYERS = 2
# N_HEADS = 8
# FORWARD_DIM = 256
# DROPOUT = 0.15
# LEARNING_RATE = 2e-4
# GRAD_CLIP = 1
# BATCH_SIZE = 16
# Optimizer: AdamW

def train(model, dataloader, optimizer, criterion, grad_clip, device):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(dataloader):
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

def evaluate(model, dataloader, device):
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0
    total_correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt = batch['tgt_output'].to(device)
            output = model(src, tgt_input)
            # print(output.shape)

            # Token-level accuracy
            pred = output.argmax(dim=-1)
            correct_tokens = (pred == tgt)
            non_zero_mask = (tgt != 0)
            masked_correct_tokens = correct_tokens & non_zero_mask
            total_correct_tokens += masked_correct_tokens.sum().item()
            total_tokens += non_zero_mask.sum().item()

            # Sequence-level accuracy
            pred_sequences = output.argmax(dim=2)
            correct_sequences = (pred_sequences == tgt).all(dim=1)
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

    vocal_in = load_vocab(vocab_in_path)
    vocal_out = load_vocab(vocab_out_path)

    Dataset_train = MyDataset(path_train, 16, vocal_in, vocal_out)
    Dataset_train.load_data()
    dataloader_train = DataLoader(Dataset_train, batch_size=16, shuffle=True)

    Dataset_test = MyDataset(path_test, 16, vocal_in, vocal_out)
    Dataset_test.load_data()
    dataloader_test = DataLoader(Dataset_test, batch_size=16, shuffle=False)

    model = Transformer(
        emb_dim=128,
        num_layers=2,
        num_heads=8,
        forward_dim=256,
        dropout=0.15,
        src_vocab_size=len(Dataset_train.vocab_in),
        tgt_vocab_size=len(Dataset_train.vocab_out),
        src_pad_idx=0,
        tgt_pad_idx=0,
    ).to(device)

    # print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    # eval

    subset = Subset(Dataset_test, range(6))
    dataloader_subset = DataLoader(subset, batch_size=2, shuffle=True)

    epoch = 10
    for i in range(epoch):
        loss = train(model, dataloader_train, optimizer, criterion, 1, device)
        token_acc, seq_acc = evaluate(model, dataloader_test, device)
        print('epoch:', i, 'loss:', loss, 'token_acc:', token_acc, 'seq_acc:', seq_acc)
    
    # save model
    torch.save(model.state_dict(), 'models/model_2.pth')


if __name__ == "__main__":
    print("Experiment 2")
    main()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    path_train = '../Data/length_split/tasks_train_length.txt'
    path_test = '../Data/length_split/tasks_test_length.txt'
    vocab_in_path = '../Data/length_split/vocab_in.json'
    vocab_out_path = '../Data/length_split/vocab_out.json'

    vocal_in = load_vocab(vocab_in_path)
    vocal_out = load_vocab(vocab_out_path)

    Dataset_train = MyDataset(path_train, 16, vocal_in, vocal_out)
    Dataset_train.load_data()
    dataloader_train = DataLoader(Dataset_train, batch_size=16, shuffle=True)

    Dataset_test = MyDataset(path_test, 16, vocal_in, vocal_out)
    Dataset_test.load_data()
    dataloader_test = DataLoader(Dataset_test, batch_size=1, shuffle=False)

    model = Transformer(
        emb_dim=128,
        num_layers=2,
        num_heads=8,
        forward_dim=256,
        dropout=0.15,
        src_vocab_size=len(Dataset_train.vocab_in),
        tgt_vocab_size=len(Dataset_train.vocab_out),
        src_pad_idx=0,
        tgt_pad_idx=0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    #load model
    model.load_state_dict(torch.load('models/model_2.pth',map_location=torch.device('cpu')))

    for i in range(10):
        Subset = MyDataset(path_test, 16, vocal_in, vocal_out)
        Subset.load_data(command_length=i)
        if len(Subset) == 0:
            continue
        dataloader_subset = DataLoader(Subset, batch_size=16, shuffle=True)
        token_acc, seq_acc = evaluate(model, dataloader_subset, device)
        print('command_length:', i, 'token_acc:', token_acc, 'seq_acc:', seq_acc)

    for i in range(20,50):
        Subset = MyDataset(path_test, 16, vocal_in, vocal_out)
        Subset.load_data(action_length=i)
        if len(Subset) == 0:
            continue
        dataloader_subset = DataLoader(Subset, batch_size=16, shuffle=True)
        token_acc, seq_acc = evaluate(model, dataloader_subset, device)
        print('action_length:', i, 'token_acc:', token_acc, 'seq_acc:', seq_acc)





