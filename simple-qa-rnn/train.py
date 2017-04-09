import os
import time
import glob
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from model import BiLSTM1
from util import get_args
import data

args = get_args()
# torch.cuda.set_device(args.gpu)

# ---- Load Datasets ------
train_file = "datasets/SimpleQuestions_v2/annotated_fb_data_train.txt"
val_file = "datasets/SimpleQuestions_v2/annotated_fb_data_valid.txt"
test_file = "datasets/SimpleQuestions_v2/annotated_fb_data_test.txt"

train_set = data.create_rp_dataset(train_file)
val_set = data.create_rp_dataset(val_file)
# test_set = data.create_rp_dataset(test_file)
dev_set = train_set[:20]  # work with few examples first

# ---- Build Vocabulary ------
w2v_map = data.load_map("resources/w2v_map_SQ.pkl")
word_to_ix = data.load_map("resources/word_to_ix_SQ.pkl")
rel_to_ix = data.load_map("resources/rel_to_ix_SQ.pkl")
vocab_size = len(word_to_ix)
num_classes = len(rel_to_ix)

# ---- Define Model, Loss, Optim ------
config = args
config.d_out = num_classes
config.n_cells = config.n_layers
if config.birnn:
    config.n_cells *= 2

model = BiLSTM1(config)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ---- Train Model ------
iter = 0
best_val_acc = -1
train_accs = []
val_accs = []
losses = [] # for plotting later
for epoch in range(args.epochs):
    # shuffle the dataset and create batches
    shuffled_indices = np.random.permutation(len(dev_set))
    batch_indices = np.split(shuffled_indices,
                             list(range(args.batch_size, len(shuffled_indices), args.batch_size)))
    epoch_loss = 0.0
    n_correct, n_total, train_acc = 0, 0, 0
    for batch_ix in batch_indices:
        batch = dev_set[batch_ix]
        batch_loss = 0.0
        for text, label in batch:
            iter += 1
            # x.shape: |S| X |D| - sentence length can vary between examples, dimension is fixed
            x = data.text_to_vector(text, w2v_map)
            y = rel_to_ix[label]

            # clear out gradients and hidden states of the model
            model.zero_grad()
            model.hidden = model.init_hidden()

            # prepare inputs for LSTM model and run forward pass
            x_var = Variable(torch.Tensor(x))
            rel_scores = model(x_var)

            # keep count train accuracy
            n_correct += (rel_scores.data.numpy().argmax() == y)
            n_total += 1

            # compute the loss, gradients, and update the parameters
            targets = Variable(torch.LongTensor( [y] ))
            loss = loss_function(rel_scores, targets)
            batch_loss += loss.data[0]
            loss.backward()
            optimizer.step()

        epoch_loss += batch_loss
        losses.append(epoch_loss)
        train_acc = (100.0 * n_correct) / n_total
        train_accs.append(train_acc)

    if ((epoch+1) % args.log_every) == 0:
        print("epoch {:4d} loss {:.3f} train_acc {:.1f}%".format(epoch+1, epoch_loss, train_acc))
