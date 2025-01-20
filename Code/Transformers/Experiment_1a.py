import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

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

    path_train = '../Data/simple_split/tasks_train_simple.txt'
    path_test = '../Data/simple_split/tasks_test_simple.txt'
    vocab_in_path = '../Data/simple_split/vocab_in.json'
    vocab_out_path = '../Data/simple_split/vocab_out.json'

    vocal_in = load_vocab(vocab_in_path)
    vocal_out = load_vocab(vocab_out_path)

    Dataset_train = MyDataset(path_train, 64, vocal_in, vocal_out)
    dataloader_train = DataLoader(Dataset_train, batch_size=64, shuffle=True)

    Dataset_test = MyDataset(path_test, 64, vocal_in, vocal_out)
    dataloader_test = DataLoader(Dataset_test, batch_size=64, shuffle=False)

    model = Transformer(
        emb_dim=128,
        num_layers=1,
        num_heads=8,
        forward_dim=512,
        dropout=0.05,
        src_vocab_size=len(Dataset_train.vocab_in),
        tgt_vocab_size=len(Dataset_train.vocab_out),
        src_pad_idx=0,
        tgt_pad_idx=0,
    ).to(device)

    # print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4)
    criterion = nn.CrossEntropyLoss()

    # eval

    subset = Subset(Dataset_test, range(6))
    dataloader_subset = DataLoader(subset, batch_size=2, shuffle=True)

    epoch = 5
    for i in range(epoch):
        loss = train(model, dataloader_train, optimizer, criterion, 1, device)
        token_acc, seq_acc = evaluate(model, dataloader_test, device)
        print('epoch:', i, 'loss:', loss, 'token_acc:', token_acc, 'seq_acc:', seq_acc)

    # save model
    # torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    main()

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('device:', device)
    # path_train = '../Data/simple_split/tasks_train_simple.txt'
    # path_test = '../Data/simple_split/tasks_test_simple.txt'
    # vocab_in_path = '../Data/simple_split/vocab_in.json'
    # vocab_out_path = '../Data/simple_split/vocab_out.json'

    # vocal_in = load_vocab(vocab_in_path)
    # vocal_out = load_vocab(vocab_out_path)

    # Dataset_train = MyDataset(path_train, 32, vocal_in, vocal_out)
    # dataloader_train = DataLoader(Dataset_train, batch_size=64, shuffle=True)

    # Dataset_test = MyDataset(path_test, 32, vocal_in, vocal_out)
    # dataloader_test = DataLoader(Dataset_test, batch_size=64, shuffle=False)

    # model = Transformer(
    #     emb_dim=128,
    #     num_layers=1,
    #     num_heads=8,
    #     forward_dim=512,
    #     dropout=0.05,
    #     src_vocab_size=len(Dataset_train.vocab_in),
    #     tgt_vocab_size=len(Dataset_train.vocab_out),
    #     src_pad_idx=0,
    #     tgt_pad_idx=0,
    # ).to(device)

    # model.load_state_dict(torch.load('model.pth'))

    # optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4)
    # criterion = nn.CrossEntropyLoss()


    # subset = Subset(Dataset_test, range(1))
    # dataloader_subset = DataLoader(subset, batch_size=1, shuffle=True)

    # for i, batch in enumerate(dataloader_subset):
    #     src = batch['src'].to(device)
    #     tgt = batch['tgt_output'].to(device)
    #     tgt_input = batch['tgt_input'].to(device)

    #     output = model(src, tgt_input)
    #     print(output.argmax(dim=-1))

    #     print(model.generate(src, tgt_input, 10))
