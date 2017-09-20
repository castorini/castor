from argparse import ArgumentParser
from tqdm import tqdm
import array
import torch
import six

def create_lookup(fname):
    fname_pt = '{}.pt'.format(fname)
    itos, vectors, dim = [], array.array('d'), None
    with open(fname, 'r') as f:
        lines = [line for line in f]
    print("Loading vectors from {}".format(fname))
    for line in tqdm(lines, total=len(lines)):
        entries = line.strip().split('\t')
        word, entries = entries[0], [float(item) for item in entries[1].split()]
        if dim is None:
            dim = len(entries)
        try:
            if isinstance(word, six.binary_type):
                word = word.decode('utf-8')
        except:
            print('non-UTF8 token', repr(word), 'ignored')
            continue
        vectors.extend(float(x) for x in entries)
        itos.append(word)

    stoi = {word: i for i, word in enumerate(itos)}
    vectors = torch.Tensor(vectors).view(-1, dim)
    print('saving vectors to', fname_pt)
    torch.save((stoi, vectors, dim), fname_pt)

if __name__ == '__main__':
    parser = ArgumentParser(description='create lookup')
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()

    create_lookup(args.input)

