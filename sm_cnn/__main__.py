import argparse
import logging
from copy import deepcopy

import os
import pprint
import random

import numpy as np
import torch

from common.dataset import DatasetFactory
from common.evaluation import EvaluatorFactory
from common.train import TrainerFactory
from utils.serialization import load_checkpoint
from sm_cnn.model import SMCNN
from sm_cnn.args import get_args


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device, keep_results=False):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device,
                                                           keep_results=keep_results)
    scores, metric_names = saved_model_evaluator.get_scores()
    logger.info('Evaluation metrics for {}'.format(split_name))
    logger.info('\t'.join([' '] + metric_names))
    logger.info('\t'.join([split_name] + list(map(str, scores))))

if __name__ == '__main__':
    # Getting args
    args = get_args()
    config = deepcopy(args)

    # Getting logger
    logger = get_logger()
    logger.info(pprint.pformat(vars(args)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != -1:
        torch.cuda.manual_seed(args.seed)

    # Dealing with device
    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        logger.info("Note: You are using GPU for training")
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        logger.info("Warning: You have Cuda but do not use it. You are using CPU for training")
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')

    if args.dataset not in ('trecqa', 'wikiqa'):
        raise ValueError('Unrecognized dataset')

    dataset_cls, embedding, train_loader, test_loader, dev_loader \
        = DatasetFactory.get_dataset(args.dataset, args.word_vectors_dir, args.word_vectors_file, args.batch_size,
                                     args.device)

    config.questions_num = dataset_cls.VOCAB_SIZE
    config.answers_num = dataset_cls.VOCAB_SIZE
    config.target_class = dataset_cls.NUM_CLASSES
    model = SMCNN(config)


    model = model.to(device)
    embedding = embedding.to(device)


    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, train_loader, args.batch_size,
                                                     args.device)
    test_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, test_loader, args.batch_size,
                                                    args.device)
    dev_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, dev_loader, args.batch_size,
                                                   args.device)


    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_interval,
        'model_outfile': args.model_outfile,
        'patience': args.patience,
        'tensorboard': args.tensorboard,
        'run_label': args.run_label,
        'logger': logger
    }

    trainer = TrainerFactory.get_trainer(args.dataset, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

    if not args.skip_training:
        total_params = 0
        for param in model.parameters():
            size = [s for s in param.size()]
            total_params += np.prod(size)
        logger.info('Total number of parameters: %s', total_params)
        model.static_question_embed.weight.data.copy_(embedding.weight)
        model.nonstatic_question_embed.weight.data.copy_(embedding.weight)
        model.static_answer_embed.weight.data.copy_(embedding.weight)
        model.nonstatic_answer_embed.weight.data.copy_(embedding.weight)

        trainer.train(args.epochs)

    _, _, state_dict, _, _ = load_checkpoint(args.model_outfile)

    for k, tensor in state_dict.items():
        state_dict[k] = tensor.to(device)

    model.load_state_dict(state_dict)
    if dev_loader:
        evaluate_dataset('dev', dataset_cls, model, embedding, dev_loader, args.batch_size, args.device)
    evaluate_dataset('test', dataset_cls, model, embedding, test_loader, args.batch_size, args.device, args.keep_results)
