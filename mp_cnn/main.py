import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from dataset import DatasetType, MPCNNDatasetFactory
from evaluation import MPCNNEvaluatorFactory
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


def train(model, optimizer, train_loader, epoch, batch_size, sample, log_interval):
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
                epoch, min(batch_idx * batch_size, len(train_loader.dataset)),
                len(train_loader.dataset) if not sample else sample,
                100. * batch_idx / (len(train_loader) if not sample else sample), loss.data[0])
            )


def test(dev_evaluator, test_evaluator):
    dev_scores, metric_names = dev_evaluator.get_scores()
    test_scores, _ = test_evaluator.get_scores()
    logger.info('Dev/Test evaluation metrics:')
    logger.info('\t'.join([' '] + metric_names))
    logger.info('\t'.join(['dev'] + list(map(str, dev_scores))))
    logger.info('\t'.join(['test'] + list(map(str, test_scores))))
    return dev_scores, test_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of Multi-Perspective CNN')
    parser.add_argument('model_outfile', help='file to save final model')
    parser.add_argument('--dataset', help='dataset to use, one of [sick, msrvid]', default='sick')
    parser.add_argument('--word-vectors-file', help='word vectors file', default=os.path.join(os.pardir, os.pardir, 'data', 'GloVe', 'glove.840B.300d.txt'))
    parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--sample', type=int, default=0, metavar='N', help='how many examples to take from each dataset, meant for quickly testing entire end-to-end pipeline (default: all)')
    parser.add_argument('--regularization', type=float, default=0.0001, metavar='REG', help='SGD regularization (default: 0.0001)')
    parser.add_argument('--max-window-size', type=int, default=3, metavar='N', help='windows sizes will be [1,max_window_size] and infinity')
    parser.add_argument('--holistic-filters', type=int, default=300, metavar='N', help='number of holistic filters')
    parser.add_argument('--per-dim-filters', type=int, default=20, metavar='N', help='number of per-dimension filters')
    parser.add_argument('--hidden-units', type=int, default=150, metavar='N', help='number of hidden units in each of the two hidden layers')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, dev_loader = MPCNNDatasetFactory.get_dataset(args.dataset, args.word_vectors_file, args.batch_size, args.cuda, args.sample)

    filter_widths = list(range(1, args.max_window_size + 1)) + [np.inf]
    model = MPCNN(300, args.holistic_filters, args.per_dim_filters, filter_widths, args.hidden_units, train_loader.dataset.num_classes)
    if args.cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.regularization)
    test_evaluator = MPCNNEvaluatorFactory.get_evaluator(args.dataset, model, test_loader, args.batch_size, args.cuda)
    dev_evaluator = MPCNNEvaluatorFactory.get_evaluator(args.dataset, model, dev_loader, args.batch_size, args.cuda)

    if not args.skip_training:
        epoch_times = []
        # TODO assuming the higher the score the better, this is not always true. Make more generic.
        best_dev_score = -1
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            logger.info('Epoch {} started...'.format(epoch))
            train(model, optimizer, train_loader, epoch, args.batch_size, args.sample, args.log_interval)
            dev_scores, test_scores = test(dev_evaluator, test_evaluator)
            end = time.time()
            duration = end - start
            logger.info('Epoch {} finished in {:.2f} minutes'.format(epoch, duration / 60))
            epoch_times.append(duration)

            if dev_scores[0] > best_dev_score:
                best_dev_score = dev_scores[0]
                torch.save(model, args.model_outfile)

        logger.info('Training took {:.2f} minutes overall...'.format(sum(epoch_times) / 60))
    else:
        model = torch.load(args.model_outfile)
        test(dev_evaluator, test_evaluator)
