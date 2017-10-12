from tqdm import tqdm
import array
import torch
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from argparse import ArgumentParser


def convert(fname, save_file):
    stoi, vectors = [], array.array('d')

    vocab_size, dim = None, None
    with open(fname, 'rb') as dim_file:
        vocab_size, dim = (int(x) for x in dim_file.readline().split())

    W = np.memmap(fname, dtype=np.double, shape=(vocab_size, dim))

    print("Loading vectors from {}".format(fname))
    vectors = []
    for line in tqdm(W, total=len(W)):
        entry = line
        vectors.extend(entry)
    vectors = torch.Tensor(vectors).view(-1, dim)

    wv = KeyedVectors.load_word2vec_format(fname, binary=True)
    stoi = {word.strip():voc.index for word, voc in wv.vocab.items()}

    print('saving vectors to', save_file)
    torch.save((stoi, vectors, dim), save_file)

if __name__ == '__main__':
    parser = ArgumentParser(description='create word embedding')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/word2vec.trecqa.pt')

    args = parser.parse_args()
    convert(args.input, args.output)
