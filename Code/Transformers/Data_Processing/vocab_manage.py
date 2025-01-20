from collections import defaultdict
import json

shared_vocab_in = defaultdict(lambda: len(shared_vocab_in))
shared_vocab_out = defaultdict(lambda: len(shared_vocab_out))

shared_vocab_in["<PAD>"], shared_vocab_in["<START>"], shared_vocab_in["<END>"] = 0, 1, 2
shared_vocab_out["<PAD>"], shared_vocab_out["<START>"], shared_vocab_out["<END>"] = 0, 1, 2

def create_vocab(train_dir, test_dir, vocab_in=None, vocab_out=None):
    vocab_in = vocab_in if vocab_in is not None else shared_vocab_in
    vocab_out = vocab_out if vocab_out is not None else shared_vocab_out
    with open(train_dir, 'r') as f:
        data = f.readlines()
        data = [line.strip('\t') for line in data]
        train_in = [line.split('IN:')[1].split('OUT:')[0].strip() for line in data]
        train_out = [line.split('OUT:')[1].strip() for line in data]

    with open(test_dir, 'r') as f:
        data = f.readlines()
        data = [line.strip('\t') for line in data]
        test_in = [line.split('IN:')[1].split('OUT:')[0].strip() for line in data]
        test_out = [line.split('OUT:')[1].strip() for line in data]

    for in_text, out_text in zip(train_in, train_out):
        for word in in_text.split():
            vocab_in[word]
        for action in out_text.split():
            vocab_out[action]

    for in_text, out_text in zip(test_in, test_out):
        for word in in_text.split():
            vocab_in[word]
        for action in out_text.split():
            vocab_out[action]

    return vocab_in, vocab_out

def save_vocab(vocab, file_path):
    with open(file_path, 'w') as f:
        json.dump(dict(vocab), f, ensure_ascii=False, indent=4)

def load_vocab(file_path):
    with open(file_path, 'r') as f:
        vocab = json.load(f)
    return defaultdict(lambda: len(vocab), vocab)

if __name__ == "__main__":
    train_dir = '../../Data/add_prim_split/tasks_train_addprim_turn_left.txt'
    test_dir = '../../Data/add_prim_split/tasks_test_addprim_turn_left.txt'
    vocab_in, vocab_out = create_vocab(train_dir, test_dir)
    save_vocab(vocab_in, '../../Data/add_prim_split/vocab_in_left.json')
    save_vocab(vocab_out, '../../Data/add_prim_split/vocab_out_left.json')
    print('vocab_in:', len(vocab_in), 'vocab_out:', len(vocab_out))
    print('vocab_in:', vocab_in, 'vocab_out:', vocab_out)
    vocab_in = load_vocab('../../Data/add_prim_split/vocab_in_left.json')
    vocab_out = load_vocab('../../Data/add_prim_split/vocab_out_left.json')
    print('vocab_in:', len(vocab_in), 'vocab_out:', len(vocab_out))
    print('vocab_in:', vocab_in, 'vocab_out:', vocab_out)