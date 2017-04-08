import os
import time
import glob
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import BiLSTM1
from util import get_args
import data

args = get_args()
# torch.cuda.set_device(args.gpu)

# ---- Load Datasets ------
train_file = "datasets/SimpleQuestions_v2/annotated_fb_data_train.txt"
valid_file = "datasets/SimpleQuestions_v2/annotated_fb_data_valid.txt"
test_file = "datasets/SimpleQuestions_v2/annotated_fb_data_test.txt"

train_set = data.create_rp_dataset(train_file)
valid_set = data.create_rp_dataset(valid_file)
# test_set = data.create_rp_dataset(test_file)
dev_set = train_set[:10]  # work with first 50 examples

# ---- Build Vocabulary ------
w2v_map = data.load_map("resources/w2v_map_SQ.pkl")
word_to_ix = data.load_map("resources/word_to_ix_SQ.pkl")
rel_to_ix = data.load_map("resources/rel_to_ix_SQ.pkl")

# ---- Train Model ------
model = BiLSTM1(args)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    shuffled_indices = np.random.permutation(len(dev_set))
    batch_indices = np.split(shuffled_indices,
                             list(range(args.batch_size, len(shuffled_indices), args.batch_size)))

    for batch_ix in batch_indices:
        batch = dev_set[batch_ix]
        for text, label in batch:
            # x.shape: |S| X |D| - sentence length can vary between examples, dimension is fixed
            x = data.text_to_vector(text, w2v_map)
            y = rel_to_ix[label]

            model.zero_grad()

            model.hidden = model.init_hidden()

            ## FIXME
            ## Get our inputs ready for the network, that is, turn them into Variables
            ## of word indices.
            # sentence_in = data.prepare_sequence(sentence, word_to_ix)
            # targets = data.prepare_sequence(tags, tag_to_ix)

            # # Step 3. Run our forward pass.
            # rel_scores = model(sentence_in)
            #
            # # Step 4. Compute the loss, gradients, and update the parameters by calling
            # # optimizer.step()
            # loss = loss_function(tag_scores, targets)
            # loss.backward()
            # optimizer.step()

