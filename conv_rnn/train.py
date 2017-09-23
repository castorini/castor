import data
import model
import numpy as np
import torch
import torch.nn as nn
import os
import random

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

def main():
    torch.cuda.set_device(1)
    set_seed(5)
    restore = False
    mbatch_size = 64
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
    optimizer = torch.optim.Adadelta(parameters, lr=0.01, weight_decay=1E-3) 
    train_set, dev_set, test_set = data_loader.load_sst_sets()

    for epoch in range(conv_rnn.epoch, 60):
        conv_rnn.train()
        optimizer.zero_grad()
        np.random.shuffle(train_set)
        print("Epoch number: {}".format(epoch))
        i = 0
        while i + mbatch_size < len(train_set):
            if i % (mbatch_size * 10) == 0:
                print("{} / {}".format(i, len(train_set)), end="\r")
            mbatch = train_set[i:i + mbatch_size]
            train_in, train_out = conv_rnn.convert_dataset(mbatch)

            train_in.cuda()
            train_out.cuda()

            scores = conv_rnn(train_in)
            loss = criterion(scores, train_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 6)
            optimizer.step()
            i += mbatch_size
            if i % 16384 == 0:
                conv_rnn.eval()
                dev_in, dev_out = conv_rnn.convert_dataset(dev_set)
                scores = conv_rnn(dev_in)
                n_correct = (torch.max(scores, 1)[1].view(len(dev_set)).data == dev_out.data).sum()
                accuracy = n_correct / len(dev_set)
                if accuracy > conv_rnn.best_dev:
                    conv_rnn.best_dev = accuracy
                    torch.save(conv_rnn, "model.pt")
                print("Dev set accuracy: {}".format(accuracy))
                conv_rnn.train()

if __name__ == "__main__":
    main()
