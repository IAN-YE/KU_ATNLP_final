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

def read_data(path_train, path_test, vocab_in_path, vocab_out_path):
    vocal_in = load_vocab(vocab_in_path)
    vocal_out = load_vocab(vocab_out_path)

    Dataset_train = MyDataset(path_train, 16, vocal_in, vocal_out)
    Dataset_train.load_data()
    dataloader_train = DataLoader(Dataset_train, batch_size=16, shuffle=True)

    Dataset_test = MyDataset(path_test, 16, vocal_in, vocal_out)
    Dataset_test.load_data()
    dataloader_test = DataLoader(Dataset_test, batch_size=16, shuffle=False)

    return Dataset_train, Dataset_test, dataloader_train, dataloader_test

nums = [1,2,4,8,16,32]
exps = [5]
print(exps)
token_res = []
seq_res = []

epoch = 5
for num in nums:
    for exp in exps:
        path_train = f'../../SCAN/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num{num}_rep{exp}.txt'
        path_test = f'../../SCAN/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num{num}_rep{exp}.txt'
        vocab_in_path = '../Data/simple_split/vocab_in.json'
        vocab_out_path = '../Data/simple_split/vocab_out.json'

        Dataset_train, Dataset_test, dataloader_train, dataloader_test = read_data(path_train, path_test, vocab_in_path, vocab_out_path)

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
    
    for i in range(epoch):
        loss = train(model, dataloader_train, optimizer, criterion, 1, device)
        token_acc, seq_acc = evaluate(model, dataloader_test, device)
        print('epoch:', i, 'loss:', loss, 'token_acc:', token_acc, 'seq_acc:', seq_acc)
        if i == epoch-1:
            token_res.append(token_acc)
            seq_res.append(seq_acc)

print(token_res)
print(seq_res)
# save model
# for i in range(epoch):
#     loss = train(model, dataloader_train, optimizer, criterion, 1, device)
#     token_acc, seq_acc = evaluate(model, dataloader_test, device)
#     print('epoch:', i, 'loss:', loss, 'token_acc:', token_acc, 'seq_acc:', seq_acc)

# torch.save(model.state_dict(), 'models/model_jump.pth')

# # turn left
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('device:', device)

# print('turn left')

# path_train = '../Data/add_prim_split/tasks_train_addprim_turn_left.txt'
# path_test = '../Data/add_prim_split/tasks_test_addprim_turn_left.txt'
# vocab_in_path = '../Data/add_prim_split/vocab_in_left.json'
# vocab_out_path = '../Data/add_prim_split/vocab_out_left.json'

# vocal_in = load_vocab(vocab_in_path)
# vocal_out = load_vocab(vocab_out_path)

# Dataset_train = MyDataset(path_train, 16, vocal_in, vocal_out)
# Dataset_train.load_data()
# dataloader_train = DataLoader(Dataset_train, batch_size=16, shuffle=True)

# Dataset_test = MyDataset(path_test, 16, vocal_in, vocal_out)
# Dataset_test.load_data()
# dataloader_test = DataLoader(Dataset_test, batch_size=16, shuffle=False)

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
# criterion = nn.CrossEntropyLoss()

# print(int(100000/len(dataloader_train)))
# # eval
# for i in range(int (100000/len(dataloader_train))):
#     loss = train(model, dataloader_train, optimizer, criterion, 1, device)
#     token_acc, seq_acc = evaluate(model, dataloader_test, device)
#     print('epoch:', i, 'loss:', loss, 'token_acc:', token_acc, 'seq_acc:', seq_acc)

# # save model
# torch.save(model.state_dict(), 'models/model_left.pth')