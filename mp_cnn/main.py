import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from dataset import DatasetType, MPCNNDatasetFactory
from model import MPCNN

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def train(model, optimizer, train_loader, epoch, sample, log_interval):
    model.train()
    for batch_idx, (sentences, labels) in enumerate(train_loader):
        sent_a, sent_b = Variable(sentences['a']), Variable(sentences['b'])
        labels = Variable(labels)
        optimizer.zero_grad()
        output = model(sent_a, sent_b)
        loss = F.kl_div(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * 1, len(train_loader) if not sample else sample,
                100. * batch_idx / (len(train_loader) if not sample else sample), loss.data[0])
            )


def test(model, loader):
    model.eval()
    test_loss = 0
    for sentences, labels in test_loader:
        sent_a, sent_b = Variable(sentences['a'], volatile=True), Variable(sentences['b'], volatile=True)
        labels = Variable(labels, volatile=True)
        output = model(sent_a, sent_b)
        test_loss += F.kl_div(output, labels, size_average=False).data[0]

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of Multi-Perspective CNN')
    parser.add_argument('--dataset', help='dataset to use, one of [sick, msrvid]', default='sick')
    parser.add_argument('--word_vectors_file', help='word vectors file', default=os.path.join(os.pardir, os.pardir, 'data', 'GloVe', 'glove.840B.300d.txt'))
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--sample', type=int, default=0, metavar='N', help='how many examples to take from each dataset, meant for quickly testing entire end-to-end pipeline (default: all)')
    parser.add_argument('--regularization', type=float, default=0.0001, metavar='REG', help='SGD regularization (default: 0.0001)')
    parser.add_argument('--holistic_filters', type=int, default=300, metavar='N', help='number of holistic filters')
    parser.add_argument('--per_dim_filters', type=int, default=20, metavar='N', help='number of per-dimension filters')
    parser.add_argument('--hidden_units', type=int, default=150, metavar='N', help='number of hidden units in each of the two hidden layers')
    parser.add_argument('--num_classes', type=int, default=5, metavar='N', help='number of classes of the label. SICK has 5 classes and MSRVID has 6.')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    train_loader, test_loader, dev_loader = MPCNNDatasetFactory.get_dataset(args.dataset, args.word_vectors_file, args.sample)

    filter_widths = [1, 2, 3, np.inf]
    model = MPCNN(300, args.holistic_filters, args.per_dim_filters, filter_widths, args.hidden_units, args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.regularization)

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_loader, epoch, args.sample, args.log_interval)
        test(model, test_loader)
