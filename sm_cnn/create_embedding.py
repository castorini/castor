from tqdm import tqdm
import array
import torch
import six
import numpy as np

from argparse import ArgumentParser


def convert(fname, vocab):
    save_file = '{}.pt'.format(fname)
    stoi, vectors, dim = [], array.array('d'), None

    with open(fname, 'rb') as f:
        vocab_size, vec_dim = [int(x.decode('utf-8')) for x in f.readline().split()]

    W = np.memmap(fname, dtype=np.double, shape=(vocab_size, vec_dim))
    print(len(W))


    print("Loading vectors from {}".format(fname))
    for line in tqdm(W, total=len(W)):
        entry = line
        if dim is None:
            dim = len(entry)

        vectors = torch.Tensor(vectors).view(-1, dim)

    with open(vocab) as f:
        stoi  = {word:i for i, word in enumerate(f)}

    print('saving vectors to', save_file)
    torch.save((stoi, vectors, dim), save_file)

if __name__ == '__main__':
    parser = ArgumentParser(description='create word embedding')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)


    args = parser.parse_args()
    convert(args.input, args.vocab)
