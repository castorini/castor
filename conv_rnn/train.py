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

def train(mbatch_size=64, n_epochs=30, restore=False, verbose=True):
    torch.cuda.set_device(0)
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
    optimizer = torch.optim.Adadelta(parameters, lr=5E-2, weight_decay=1E-3)
    #optimizer = torch.optim.SGD(parameters, lr=5E-5, weight_decay=1E-3, nesterov=True, momentum=1E-4)
    train_set, dev_set, test_set = data_loader.load_sst_sets()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.33)

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
            torch.nn.utils.clip_grad_norm(parameters, 7)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    dev_acc = train(mbatch_size=64, n_epochs=30, verbose=True, restore=False)

