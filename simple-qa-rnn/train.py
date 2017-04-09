import os
import time
import glob
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from model import BiLSTM
from util import get_args
import data

args = get_args()
# torch.cuda.set_device(args.gpu)


# ---- Helper Methods ------
def evaluate_dataset(data_set, model, w2v_map, label_to_ix):
    n_total = len(data_set)
    n_correct = 0
    for sentence, label in data_set:
        inputs, targets = data.create_tensorized_data(sentence, label, w2v_map, label_to_ix)
        scores = model(inputs)
        pred_label_ix = scores.data.numpy()[0].argmax()
        correct_label_ix = label_to_ix[label]
        # print("pred_label: {}, correct_label: {}".format(pred_label_ix, correct_label_ix))
        n_correct += (pred_label_ix == correct_label_ix)
    acc = (100.0 * n_correct) / n_total
    return acc

# ---- Load Datasets ------
train_file = "datasets/SimpleQuestions_v2/annotated_fb_data_train.txt"
val_file = "datasets/SimpleQuestions_v2/annotated_fb_data_valid.txt"
test_file = "datasets/SimpleQuestions_v2/annotated_fb_data_test.txt"

train_set = data.create_rp_dataset(train_file)
val_set = data.create_rp_dataset(val_file)
# test_set = data.create_rp_dataset(test_file)
dev_set = train_set[:20] # work with few examples first

# ---- Build Vocabulary ------
w2v_map = data.load_map("resources/w2v_map_SQ.pkl")
word_to_ix = data.load_map("resources/word_to_ix_SQ.pkl")
label_to_ix = data.load_map("resources/rel_to_ix_SQ.pkl")
vocab_size = len(word_to_ix)
num_classes = len(label_to_ix)

# ---- Define Model, Loss, Optim ------
config = args
config.d_out = num_classes
# config.n_layers = 2
model = BiLSTM(config)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ---- Train Model ------
iter = 0
for epoch in range(args.epochs):
    # shuffle the dataset and create batches
    shuffled_indices = np.random.permutation(len(train_set))
    batch_indices = np.split(shuffled_indices,
                             list(range(args.batch_size, len(shuffled_indices), args.batch_size)))
    epoch_loss = 0.0
    for batch_ix in batch_indices:
        batch = train_set[batch_ix]
        for sentence, label in batch:
            iter += 1
            inputs, targets = data.create_tensorized_data(sentence, label, w2v_map, label_to_ix)

            # clear out gradients and hidden states of the model
            model.zero_grad()
            model.hidden = model.init_hidden()

            # prepare inputs for LSTM model and run forward pass
            scores = model(inputs)

            # compute the loss, gradients, and update the parameters
            loss = loss_function(scores, targets)
            epoch_loss += loss.data[0]
            loss.backward()
            optimizer.step()

    # log after every epoch
    val_acc = evaluate_dataset(val_set, model, w2v_map, label_to_ix)
    print("epoch {:2d}, loss {:6f}, val_acc {:.1f}%".format(epoch+1, epoch_loss, val_acc))

