import argparse
import os

import numpy as np
import torch

import preprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of Multi-Perspective CNN')
    parser.add_argument('--dataset_folder', help='directory containing train, dev, test sets', default=os.path.join(os.pardir, os.pardir, 'data', 'sick'))
    parser.add_argument('--word_vectors_file', help='word vectors file', default=os.path.join(os.pardir, os.pardir, 'data', 'GloVe', 'glove.840B.300d.txt'))
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    word_index, embedding = preprocessing.get_glove_embedding(args.word_vectors_file, args.dataset_folder)
