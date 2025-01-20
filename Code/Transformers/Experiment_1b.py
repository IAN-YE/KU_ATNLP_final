import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import itertools

from Data_Processing.dataloader import MyDataset
from Transformer.Transformer import Transformer

from Experiment_1a import train, evaluate
from Data_Processing.vocab_manage import load_vocab

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path_train = '../Data/simple_split/tasks_train_simple.txt'
path_test = '../Data/simple_split/tasks_test_simple.txt'
vocab_in_path = '../Data/simple_split/vocab_in.json'
vocab_out_path = '../Data/simple_split/vocab_out.json'

vocal_in = load_vocab(vocab_in_path)
vocal_out = load_vocab(vocab_out_path)

Dataset_train = MyDataset(path_train, 64, vocal_in, vocal_out)
Dataset_train.load_data()
dataloader_train = DataLoader(Dataset_train, batch_size=64, shuffle=True)

Dataset_test = MyDataset(path_test, 64, vocal_in, vocal_out)
Dataset_test.load_data()
dataloader_test = DataLoader(Dataset_test, batch_size=64, shuffle=False)

print(Dataset_train.n_samples,Dataset_test.n_samples)

data_percentage = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1]
epoches = 100000
for i in data_percentage:
    print('data_percentage:', i)
    subset = Subset(Dataset_train, range(int(len(Dataset_train)*i)))
    dataloader_train = DataLoader(subset, batch_size=64, shuffle=True)
    print(len(Dataset_train), len(subset))
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4)
    criterion = nn.CrossEntropyLoss()

    print(int(epoches/len(subset)))
    e = int (epoches/len(subset))
    for epoch in range(int(epoches/len(subset))):
        train(model, dataloader_train, optimizer, criterion, 1, device)
        token_acc, seq_acc = evaluate(model, dataloader_test, device)
        if epoch == e-1:
            print('epoch:', i, 'token_acc:', token_acc, 'seq_acc:', seq_acc)
