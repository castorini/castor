import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import data
import model

class RandomSearch(object):
    def __init__(self, params):
        self.params = params

    def __iter__(self):
        param_space = list(GridSearch(self.params))
        random.shuffle(param_space)
        for param in param_space:
            yield param

class GridSearch(object):
    def __init__(self, params):
        self.params = params
        self.param_lengths = [len(param) for param in self.params]
        self.indices = [1] * len(params)

    def _update(self, carry_idx):
        if carry_idx >= len(self.params):
            return True
        if self.indices[carry_idx] < self.param_lengths[carry_idx]:
            self.indices[carry_idx] += 1
            return False
        else:
            self.indices[carry_idx] = 1
            return False or self._update(carry_idx + 1)

    def __iter__(self):
        self.stop_next = False
        self.indices = [1] * len(self.params)
        return self

    def __next__(self):
        if self.stop_next:
            raise StopIteration
        result = [param[idx - 1] for param, idx in zip(self.params, self.indices)]
        self.indices[0] += 1
        if self.indices[0] == self.param_lengths[0] + 1:
            self.indices[0] = 1
            self.stop_next = self._update(1)
        return result

def train(**kwargs):
    mbatch_size = kwargs["mbatch_size"]
    n_epochs = kwargs["n_epochs"]
    restore = kwargs["restore"]
    verbose = not kwargs["quiet"]
    lr = kwargs["lr"]
    weight_decay = kwargs["weight_decay"]
    gradient_clip = kwargs["gradient_clip"]
    seed = kwargs["seed"]

    if not kwargs["no_cuda"]:
        torch.cuda.set_device(1)
    model.set_seed(seed)
    data_loader = data.SSTDataLoader("data")
    if restore:
        conv_rnn = torch.load(kwargs["input_file"])
    else:
        id_dict, weights, unk_vocab_list = data_loader.load_embed_data()
        word_model = model.SSTWordEmbeddingModel(id_dict, weights, unk_vocab_list)
        if not kwargs["no_cuda"]:
            word_model.cuda()
        conv_rnn = model.ConvRNNModel(word_model, **kwargs)
        if not kwargs["no_cuda"]:
            conv_rnn.cuda()

    criterion = nn.CrossEntropyLoss()
    parameters = list(filter(lambda p: p.requires_grad, conv_rnn.parameters()))
    optimizer = torch.optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
    train_set, dev_set, test_set = data_loader.load_sst_sets()
    best_dev = 0

    for epoch in range(n_epochs):
        conv_rnn.train()
        optimizer.zero_grad()
        np.random.shuffle(train_set)
        print("Epoch number: {}".format(epoch), end="\r")
        if verbose:
            print()
        i = 0
        while i + mbatch_size < len(train_set):
            if verbose and i % (mbatch_size * 10) == 0:
                print("{} / {}".format(i, len(train_set)), end="\r")
            mbatch = train_set[i:i + mbatch_size]
            train_in, train_out = conv_rnn.convert_dataset(mbatch)

            if not kwargs["no_cuda"]:
                train_in.cuda()
                train_out.cuda()

            scores = conv_rnn(train_in)
            loss = criterion(scores, train_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, gradient_clip)
            optimizer.step()
            i += mbatch_size
            if i % (mbatch_size * 256) == 0:
                conv_rnn.eval()
                dev_in, dev_out = conv_rnn.convert_dataset(dev_set)
                scores = conv_rnn(dev_in)
                n_correct = (torch.max(scores, 1)[1].view(len(dev_set)).data == dev_out.data).sum()
                accuracy = n_correct / len(dev_set)
                if accuracy > best_dev:
                    best_dev = accuracy
                    torch.save(conv_rnn, kwargs["output_file"])
                if verbose:
                    print("Dev set accuracy: {}".format(accuracy))
                conv_rnn.train()
    return best_dev

def do_random_search():
    test_grid = [[0.15, 0.2], [4, 5, 6], [10, 20], [150, 200], [3, 4, 5], [200, 300], [200, 250]]
    max_params = None
    max_acc = 0.
    for args in RandomSearch(test_grid):
        sf, gc, ptce, hid, seed, fc_size, fmaps = args
        print("Testing {}".format(args))
        dev_acc = train(mbatch_size=64, n_epochs=7, verbose=False, restore=False, gradient_clip=gc,
                schedule_factor=sf, patience=ptce, hidden_size=hid, seed=seed, n_feature_maps=fmaps, 
                fc_size=fc_size)
        print("Dev accuracy: {}".format(dev_acc))
        if dev_acc > max_acc:
            print("Found current max")
            max_acc = dev_acc
            max_params = args
    print("Best params: {}".format(max_params))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout_prob", default=0.5, type=float)
    parser.add_argument("--fc_size", default=200, type=int)
    parser.add_argument("--gpu_number", default=0, type=int)
    parser.add_argument("--gradient_clip", default=5, type=float)
    parser.add_argument("--hidden_size", default=200, type=int)
    parser.add_argument("--input_file", default="saves/model.pt", type=str)
    parser.add_argument("--lr", default=5E-2, type=float)
    parser.add_argument("--mbatch_size", default=64, type=int)
    parser.add_argument("--n_epochs", default=30, type=int)
    parser.add_argument("--n_feature_maps", default=200, type=float)
    parser.add_argument("--n_labels", default=5, type=int)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--output_file", default="saves/model.pt", type=str)
    parser.add_argument("--random_search", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--rnn_type", choices=["lstm", "gru"], default="lstm", type=str)
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--weight_decay", default=1E-3, type=float)
    args = parser.parse_args()
    if args.random_search:
        do_random_search()
        return
    train(**vars(args))

if __name__ == "__main__":
    main()

