import argparse
import os

import numpy as np
import torch

from dataset import DatasetType, MPCNNDataset
import preprocessing


def train(train_loader, epochs):
    for batch_idx, (sentences, labels) in enumerate(train_loader):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of Multi-Perspective CNN')
    parser.add_argument('--dataset_folder', help='directory containing train, dev, test sets', default=os.path.join(os.pardir, os.pardir, 'data', 'sick'))
    parser.add_argument('--word_vectors_file', help='word vectors file', default=os.path.join(os.pardir, os.pardir, 'data', 'GloVe', 'glove.840B.300d.txt'))
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    word_index, embedding = preprocessing.get_glove_embedding(args.word_vectors_file, args.dataset_folder)

    train_loader = torch.utils.data.DataLoader(MPCNNDataset(args.dataset_folder, DatasetType.TRAIN, word_index, embedding), batch_size=1)

    train(train_loader, args.epochs)
