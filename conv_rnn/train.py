import argparse
import data
import model
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random

FLAGS = None

def set_seed(seed=0):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def clip_weights(parameter, s=3):
    norm = parameter.weight.data.norm()
    if norm < s:
        return
    parameter.weight.data.mul_(s / norm)

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
    mbatch_size = kwargs.get("mbatch_size", 64)
    n_epochs = kwargs.get("n_epochs", 30)
    restore = kwargs.get("restore", False)
    verbose = kwargs.get("verbose", True)
    lr = kwargs.get("lr", 5E-2)
    weight_decay = kwargs.get("weight_decay", 1E-3)
    schedule_factor = kwargs.get("schedule_factor", 0.4)
    gradient_clip = kwargs.get("gradient_clip", 6)

    torch.cuda.set_device(1)
    set_seed(5)
    data_loader = data.SSTDataLoader("data")
    if restore:
        conv_rnn = torch.load("model.pt")
    else:
        id_dict, weights, unk_vocab_list = data_loader.load_embed_data()
        word_model = model.SSTWordEmbeddingModel(id_dict, weights, unk_vocab_list)
        word_model.cuda()
        conv_rnn = model.ConvRNNModel(word_model, batch_size=mbatch_size)
        conv_rnn.cuda()

    criterion = nn.CrossEntropyLoss()
    parameters = list(filter(lambda p: p.requires_grad, conv_rnn.parameters()))
    optimizer = torch.optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(parameters, lr=5E-5, weight_decay=1E-3, nesterov=True, momentum=1E-4)
    train_set, dev_set, test_set = data_loader.load_sst_sets()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=schedule_factor)

    for epoch in range(conv_rnn.epoch, n_epochs):
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

            train_in.cuda()
            train_out.cuda()

            scores = conv_rnn(train_in)
            loss = criterion(scores, train_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, gradient_clip)
            optimizer.step()
            i += mbatch_size
            if i % 16384 == 0:
                conv_rnn.eval()
                dev_in, dev_out = conv_rnn.convert_dataset(dev_set)
                scores = conv_rnn(dev_in)
                n_correct = (torch.max(scores, 1)[1].view(len(dev_set)).data == dev_out.data).sum()
                accuracy = n_correct / len(dev_set)
                scheduler.step(accuracy)
                if accuracy > conv_rnn.best_dev:
                    conv_rnn.best_dev = accuracy
                    torch.save(conv_rnn, "saves/model.pt")
                if verbose:
                    print("Dev set accuracy: {}".format(accuracy))
                conv_rnn.train()
    return conv_rnn.best_dev

if __name__ == "__main__":
    test_grid = [[0.6, 0.4, 0.2], [6, 7, 8], [1E-2, 1E-3, 1E-4]]
    #test_grid=[[0.6],[6],[1E-3]]
    max_params = None
    max_acc = 0.
    for schedule_factor, gradient_clip, weight_decay in GridSearch(test_grid):
        dev_acc = train(mbatch_size=64, n_epochs=7, verbose=False, restore=False,
                gradient_clip=gradient_clip, schedule_factor=schedule_factor, weight_decay=weight_decay)
        if dev_acc > max_acc:
            max_acc = dev_acc
            max_params = [schedule_factor, gradient_clip, weight_decay]
    print(max_params)

