## The data loader class for the dataset

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch
from .vocab_manage import shared_vocab_in, shared_vocab_out, load_vocab

class MyDataset(Dataset):
    def __init__(self, data_dir, batch_size, vocab_in, vocab_out, shuffle=True, max_len=64):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.vocab_in = vocab_in if vocab_in is not None else shared_vocab_in
        self.vocab_out = vocab_out if vocab_out is not None else shared_vocab_out
        self.max_len = max_len

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = None
        self.tgt_input = None
        self.tgt_output = None
        self.n_samples = None
        self.n_batches = None
        self.batch_index = 0

    def load_data(self, action_length=None, command_length=None):
        with open(self.data_dir, 'r') as f:
            data = f.readlines()
            data = [line.strip() for line in data]

        data_in = [line.split('IN:')[1].split('OUT:')[0].strip() for line in data]
        data_out = [line.split('OUT:')[1].strip() for line in data]

        self.data = []
        for in_text, out_text in zip(data_in, data_out):
            in_indices = [self.vocab_in.get(word, self.vocab_in["<PAD>"]) for word in in_text.split()]
            in_indices = in_indices[:self.max_len - 1] + [self.vocab_in["<END>"]]

            out_indices = [self.vocab_out["<START>"]] + [self.vocab_out.get(action, self.vocab_out["<PAD>"]) for action
                                                         in out_text.split()] + [self.vocab_out["<END>"]]

            if command_length is not None:
                if len(in_indices) - 1 == command_length:
                    in_indices += [self.vocab_in["<PAD>"]] * (self.max_len - len(in_indices))
                else:
                    continue

            if action_length is not None:
                if len(out_indices) - 2 == action_length:
                    out_indices += [self.vocab_out["<PAD>"]] * (self.max_len - len(out_indices))
                else:
                    continue

            tgt_input = out_indices[:-1]
            tgt_output = out_indices[1:]
            in_indices += [self.vocab_in["<PAD>"]] * (self.max_len - len(in_indices))
            out_indices += [self.vocab_out["<PAD>"]] * (self.max_len - len(out_indices))

            tgt_input += [self.vocab_out["<PAD>"]] * (self.max_len - len(tgt_input))
            tgt_output += [self.vocab_out["<PAD>"]] * (self.max_len - len(tgt_output))

            self.data.append({
                "src": torch.tensor(in_indices, dtype=torch.long),
                "tgt_input": torch.tensor(tgt_input, dtype=torch.long),
                "tgt_output": torch.tensor(tgt_output, dtype=torch.long),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# path_train = '../../Data/simple_split/tasks_train_simple.txt'
# path_test = '../../Data/simple_split/tasks_test_simple.txt'
# vocab_in_path = '../../Data/simple_split/vocab_in.json'
# vocab_out_path = '../../Data/simple_split/vocab_out.json'

# vocal_in = load_vocab(vocab_in_path)
# vocal_out = load_vocab(vocab_out_path)

# Dataset_train = MyDataset(path_train, 32, vocal_in, vocal_out)
# Dataset_train.load_data()
# dataloader_train = DataLoader(Dataset_train, batch_size=1, shuffle=True)

# for i, batch in enumerate(dataloader_train):
#     print(batch['src'])
#     print(batch['tgt_input'])
#     print(batch['tgt_output'])
#     break

# Dataset_test = MyDataset(path_test, 32, vocal_in, vocal_out)
# Dataset_test.load_data(action_length=10)
# dataloader_test = DataLoader(Dataset_test, batch_size=2, shuffle=False)
#
# for i, batch in enumerate(dataloader_test):
#     print(batch['src'])
#     print(batch['tgt_input'])
#     print(batch['tgt_output'])
#     break

