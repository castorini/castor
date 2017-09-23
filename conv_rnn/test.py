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
# Best dev: 0.5140781108083561
# Test: 0.4737
def main():
    torch.cuda.set_device(1)
    set_seed(5)
    data_loader = data.SSTDataLoader("data")
    conv_rnn = torch.load("model.pt")
    conv_rnn.cuda()
    _, _, test_set = data_loader.load_sst_sets()

    conv_rnn.eval()
    test_in, test_out = conv_rnn.convert_dataset(test_set)
    scores = conv_rnn(test_in)
    n_correct = (torch.max(scores, 1)[1].view(len(test_set)).data == test_out.data).sum()
    accuracy = n_correct / len(test_set)
    print("Test set accuracy: {}".format(accuracy))

if __name__ == "__main__":
    main()
