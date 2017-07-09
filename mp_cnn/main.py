import argparse
import os

import numpy as np
import torch
from torch.autograd import Variable

from dataset import DatasetType, MPCNNDataset
from model import MPCNN
import preprocessing


# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def train(model, train_loader, epochs):
    for batch_idx, (sentences, labels) in enumerate(train_loader):
        sent_a, sent_b = sentences['a'], sentences['b']
        output = model(Variable(sent_a), Variable(sent_b))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of Multi-Perspective CNN')
    parser.add_argument('--dataset_folder', help='directory containing train, dev, test sets', default=os.path.join(os.pardir, os.pardir, 'data', 'sick'))
    parser.add_argument('--word_vectors_file', help='word vectors file', default=os.path.join(os.pardir, os.pardir, 'data', 'GloVe', 'glove.840B.300d.txt'))
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--holistic_filters', type=int, default=300, metavar='N', help='number of holistic filters')
    parser.add_argument('--per_dim_filters', type=int, default=20, metavar='N', help='number of per-dimension filters')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    word_index, embedding = preprocessing.get_glove_embedding(args.word_vectors_file, args.dataset_folder)
    logger.info('Finished loading GloVe embedding for vocab in data...')

    train_loader = torch.utils.data.DataLoader(MPCNNDataset(args.dataset_folder, DatasetType.TRAIN, word_index, embedding), batch_size=1)

    filter_widths = [1, 2, 3]
    model = MPCNN(300, args.holistic_filters, args.per_dim_filters, filter_widths)
    train(model, train_loader, args.epochs)
