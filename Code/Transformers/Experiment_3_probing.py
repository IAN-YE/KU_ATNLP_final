import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset

from Data_Processing.dataloader import MyDataset
from Data_Processing.vocab_manage import load_vocab
from Transformer.Transformer import Transformer

from Experiment_1a import train, evaluate
from torch.utils.data import DataLoader, Subset

epoch = 10
# jump
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

print('jump')

path_train = '../Data/add_prim_split/tasks_train_addprim_jump.txt'
path_test = '../Data/add_prim_split/tasks_test_addprim_jump.txt'
path_select = '../Data/add_prim_split/select_jump_cases.txt'
vocab_in_path = '../Data/add_prim_split/vocab_in_jump.json'
vocab_out_path = '../Data/add_prim_split/vocab_out_jump.json'

vocal_in = load_vocab(vocab_in_path)
vocal_out = load_vocab(vocab_out_path)

Dataset_train = MyDataset(path_train, 16, vocal_in, vocal_out)
Dataset_train.load_data()
dataloader_train = DataLoader(Dataset_train, batch_size=16, shuffle=True)

Dataset_test = MyDataset(path_test, 16, vocal_in, vocal_out)
Dataset_test.load_data()
dataloader_test = DataLoader(Dataset_test, batch_size=16, shuffle=False)




# eval

# merge subset with train
select_num = range(3,48)


epoch = 5
print('command_length----')
for num in select_num:
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
    print('select:', num)
    Dataset_select = MyDataset(path_test, 16, vocal_in, vocal_out)
    Dataset_select.load_data(command_length=num)
    if len(Dataset_select) < 4:
        continue 
    subset = Subset(Dataset_select, range(4))
    print(len(Dataset_train), len(subset))
    new_data = ConcatDataset([subset, Dataset_train])
    dataloader_subset = DataLoader(new_data, batch_size=16, shuffle=True)
    for i in range(epoch):
        loss = train(model, dataloader_subset, optimizer, criterion, 1, device)
        token_acc, seq_acc = evaluate(model, dataloader_test, device)
        print('epoch:', i, 'loss:', loss, 'token_acc:', token_acc, 'seq_acc:', seq_acc)
