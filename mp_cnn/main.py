import argparse
import os

import numpy as np
import torch
import torch.optim as optim

from dataset import MPCNNDatasetFactory
from evaluation import MPCNNEvaluatorFactory
from model import MPCNN
from train import MPCNNTrainerFactory

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of Multi-Perspective CNN')
    parser.add_argument('model_outfile', help='file to save final model')
    parser.add_argument('--dataset', help='dataset to use, one of [sick, msrvid]', default='sick')
    parser.add_argument('--word-vectors-file', help='word vectors file', default=os.path.join(os.pardir, os.pardir, 'data', 'GloVe', 'glove.840B.300d.txt'))
    parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training (default: false)')
    parser.add_argument('--sparse-features', action='store_true', default=False, help='use sparse features (default: false)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use: adam or sgd (default: adam)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr-reduce-factor', type=float, default=0.3, help='learning rate reduce factor after plateau (default: 0.3)')
    parser.add_argument('--patience', type=float, default=2, help='learning rate patience after seeing plateau (default: 2)')
    parser.add_argument('--momentum', type=float, default=0, help='momentum (default: 0)')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Adam epsilon (default: 1e-8)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--sample', type=int, default=0, help='how many examples to take from each dataset, meant for quickly testing entire end-to-end pipeline (default: all)')
    parser.add_argument('--regularization', type=float, default=0.0001, help='Regularization for the optimizer (default: 0.0001)')
    parser.add_argument('--max-window-size', type=int, default=3, help='windows sizes will be [1,max_window_size] and infinity (default: 300)')
    parser.add_argument('--holistic-filters', type=int, default=300, help='number of holistic filters (default: 300)')
    parser.add_argument('--per-dim-filters', type=int, default=20, help='number of per-dimension filters (default: 20)')
    parser.add_argument('--hidden-units', type=int, default=150, help='number of hidden units in each of the two hidden layers (default: 150)')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='use TensorBoard to visualize training (default: false)')
    parser.add_argument('--run-label', type=str, help='label to describe run')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, dev_loader = MPCNNDatasetFactory.get_dataset(args.dataset, args.word_vectors_file, args.batch_size, args.cuda, args.sample)

    filter_widths = list(range(1, args.max_window_size + 1)) + [np.inf]
    input_channels = 300
    model = MPCNN(input_channels, args.holistic_filters, args.per_dim_filters, filter_widths, args.hidden_units, train_loader.dataset.num_classes, args.dropout, args.sparse_features)
    if args.cuda:
        model.cuda()
    optimizer = None
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization, eps=args.epsilon)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.regularization)
    else:
        raise ValueError('optimizer not recognized: it should be either adam or sgd')
    train_evaluator = MPCNNEvaluatorFactory.get_evaluator(args.dataset, model, train_loader, args.batch_size, args.cuda)
    test_evaluator = MPCNNEvaluatorFactory.get_evaluator(args.dataset, model, test_loader, args.batch_size, args.cuda)
    dev_evaluator = MPCNNEvaluatorFactory.get_evaluator(args.dataset, model, dev_loader, args.batch_size, args.cuda)

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'sample': args.sample,
        'log_interval': args.log_interval,
        'model_outfile': args.model_outfile,
        'lr_reduce_factor': args.lr_reduce_factor,
        'patience': args.patience,
        'tensorboard': args.tensorboard,
        'run_label': args.run_label
    }
    trainer = MPCNNTrainerFactory.get_trainer(args.dataset, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

    if not args.skip_training:
        total_params = 0
        for param in model.parameters():
            size = [s for s in param.size()]
            total_params += np.prod(size)
        logger.info('Total number of parameters: %s', total_params)
        trainer.train(args.epochs)

    model = torch.load(args.model_outfile)
    test_evaluator = MPCNNEvaluatorFactory.get_evaluator(args.dataset, model, test_loader, args.batch_size, args.cuda)
    scores, metric_names = test_evaluator.get_scores()
    logger.info('Evaluation metrics for test')
    logger.info('\t'.join([' '] + metric_names))
    logger.info('\t'.join(['test'] + list(map(str, scores))))
