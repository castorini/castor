import os
import sys
import time
import glob
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from model import BiLSTM
from args import get_args
import data
from utils.vocab import Vocab
from utils.read_data import *

args = get_args()
# torch.cuda.set_device(args.gpu)

# ---- dataset paths ------
data_dir = "data/SimpleQuestions_v2/"
train_file = os.path.join(data_dir, "annotated_fb_data_train.txt")
val_file = os.path.join(data_dir, "annotated_fb_data_valid.txt")
test_file = os.path.join(data_dir, "annotated_fb_data_test.txt")

# ---- load GloVe embeddings ------
embed_pt_filepath = 'data/glove/glove.840B.300d.pt'
emb_w2i, emb_vecs = read_embedding(embed_pt_filepath)
emb_vocab = Vocab(emb_w2i)
emb_dim = emb_vecs.size()[1]

# ---- create dataset vocabulary and embeddings ------
vocab_pt_filepath = os.path.join(data_dir, "vocab.pt")
word2index_dict, rel2index_dict = torch.load(vocab_pt_filepath)

vocab_size = len(word2index_dict)
num_classes = len(rel2index_dict)
print('vocab size = {}'.format(vocab_size))
print('num classes = {}'.format(num_classes))

word_vocab = Vocab(word2index_dict)
word_vocab.add_unk_token("<UNK>")
rel_vocab = Vocab(rel2index_dict)

num_unk = 0
vecs = torch.FloatTensor(vocab_size, emb_dim)
for i in range(vocab_size):
    word = word_vocab.get_token(i)
    if emb_vocab.contains(word):
        vecs[i] = emb_vecs[emb_vocab.get_index(word)]
    else:
        num_unk += 1
        vecs[i].uniform_(-0.05, 0.05)

print('unk vocab count = {}'.format(num_unk))
emb_vocab = None
emb_vecs = None

# ---- load datasets ------
train_dataset = read_dataset(train_file, word_vocab, rel_vocab)
val_dataset = read_dataset(val_file, word_vocab, rel_vocab)
test_dataset = read_dataset(test_file, word_vocab, rel_vocab)
print('train_file: {}, num train = {}'.format(train_file, train_dataset["size"]))
print('val_file: {}, num dev   = {}'.format(val_file, val_dataset["size"]))
print('test_file: {}, num test  = {}'.format(test_file, test_dataset["size"]))


# ---- Define Model, Loss, Optim ------
config = args
config.d_out = num_classes
config.n_directions = 2 if config.birnn else 1
print(config)
