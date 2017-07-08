from enum import Enum
import os

import torch
from torch.autograd import Variable
import torch.utils.data as data


class DatasetType(Enum):
    TRAIN = 1
    TEST = 2
    DEV = 3


class MPCNNDataset(data.Dataset):
    train_folder = 'train'
    test_folder = 'test'
    dev_folder = 'dev'

    def __init__(self, dataset_root, dataset_type, word_index, embedding):
        if not isinstance(dataset_type, DatasetType):
            raise ValueError('dataset_type ({}) must be of type DatasetType enum'.format(dataset_type))

        if dataset_type == DatasetType.TRAIN:
            subfolder = MPCNNDataset.train_folder
        elif dataset_type == DatasetType.TEST:
            subfolder = MPCNNDataset.test_folder
        else:
            subfolder = MPCNNDataset.dev_folder

        dataset_dir = os.path.join(dataset_root, subfolder)
        if not os.path.exists(dataset_dir):
            raise RuntimeError('{} does not exist'.format(dataset_dir))

        sent_a = self._load(dataset_dir, 'a.txt')
        sent_b = self._load(dataset_dir, 'b.txt')
        self.sentences = []
        for i in range(len(sent_a)):
            sent_pair = {}
            sent_pair['a'] = self._get_sentence_embeddings(sent_a[i], word_index, embedding)
            sent_pair['b'] = self._get_sentence_embeddings(sent_b[i], word_index, embedding)
            self.sentences.append(sent_pair)
        self.labels = self._load(dataset_dir, 'sim.txt', float)

    def _load(self, dataset_dir, fname, type_converter=str):
        data = []
        with open(os.path.join(dataset_dir, fname), 'r') as f:
            for line in f:
                stripped_line = line.rstrip()
                item = type_converter(stripped_line)
                data.append(item)
        return data

    def _get_sentence_embeddings(self, sentence, word_index, embedding):
        tokens = sentence.split(' ')
        sentence_embedding = torch.zeros(300, len(tokens))
        found_pos, found_emb_idx = [], []
        for i, token in enumerate(tokens):
            if token in word_index:
                found_pos.append(i)
                found_emb_idx.append(word_index[token])

        found_word_vecs = embedding(Variable(torch.LongTensor(found_emb_idx)))
        # TODO Handle unknown word vector
        for i, v in enumerate(found_pos):
            sentence_embedding[:, v] = found_word_vecs[i].data
        return sentence_embedding

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
