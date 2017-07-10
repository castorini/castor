from enum import Enum
import math
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data

import preprocessing

# logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class DatasetType(Enum):
    TRAIN = 1
    TEST = 2
    DEV = 3


class MPCNNDatasetFactory(object):
    """
    Get the corresponding Dataset class for a particular dataset.
    """
    @staticmethod
    def get_dataset(dataset_name, word_vectors_file, sample):
        extra_args = {}
        if sample:
            sample_indices = list(range(sample))
            subset_random_sampler = data.sampler.SubsetRandomSampler(sample_indices)
            extra_args['sampler'] = subset_random_sampler
        if dataset_name == 'sick':
            train_loader = torch.utils.data.DataLoader(SICKDataset(DatasetType.TRAIN), batch_size=1, **extra_args)
            test_loader = torch.utils.data.DataLoader(SICKDataset(DatasetType.TEST), batch_size=1, **extra_args)
            dev_loader = torch.utils.data.DataLoader(SICKDataset(DatasetType.DEV), batch_size=1, **extra_args)
        elif dataset_name == 'msrvid':
            raise NotImplementedError('msrvid Dataset is not yet implemented.')
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))

        word_index, embedding = preprocessing.get_glove_embedding(word_vectors_file, train_loader.dataset.dataset_root)
        logger.info('Finished loading GloVe embedding for vocab in data...')

        train_loader.dataset.initialize(word_index, embedding)
        test_loader.dataset.initialize(word_index, embedding)
        dev_loader.dataset.initialize(word_index, embedding)
        return train_loader, test_loader, dev_loader


class MPCNNDataset(data.Dataset):
    train_folder = 'train'
    test_folder = 'test'
    dev_folder = 'dev'
    dataset_root = None  # subclass will override this

    def __init__(self, dataset_type):
        if not isinstance(dataset_type, DatasetType):
            raise ValueError('dataset_type ({}) must be of type DatasetType enum'.format(dataset_type))

        if dataset_type == DatasetType.TRAIN:
            subfolder = MPCNNDataset.train_folder
        elif dataset_type == DatasetType.TEST:
            subfolder = MPCNNDataset.test_folder
        else:
            subfolder = MPCNNDataset.dev_folder

        self.dataset_dir = os.path.join(self.dataset_root, subfolder)
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError('{} does not exist'.format(self.dataset_dir))

    def initialize(self, word_index, embedding):
        """
        Convert sentences into sentence embeddings.
        """
        sent_a = self._load(self.dataset_dir, 'a.txt')
        sent_b = self._load(self.dataset_dir, 'b.txt')
        self.sentences = []
        for i in range(len(sent_a)):
            sent_pair = {}
            sent_pair['a'] = self._get_sentence_embeddings(sent_a[i], word_index, embedding)
            sent_pair['b'] = self._get_sentence_embeddings(sent_b[i], word_index, embedding)
            self.sentences.append(sent_pair)
        self.labels = self._load(self.dataset_dir, 'sim.txt', float)

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
        sentence_embedding = torch.Tensor(300, len(tokens)).normal_(0, 1)
        found_pos, found_emb_idx = [], []
        for i, token in enumerate(tokens):
            if token in word_index:
                found_pos.append(i)
                found_emb_idx.append(word_index[token])

        found_word_vecs = embedding(Variable(torch.LongTensor(found_emb_idx)))
        for i, v in enumerate(found_pos):
            sentence_embedding[:, v] = found_word_vecs[i].data
        return sentence_embedding

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class SICKDataset(MPCNNDataset):

    dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'sick')

    def __init__(self, dataset_type):
        super(SICKDataset, self).__init__(dataset_type)

    def initialize(self, word_index, embedding):
        super(SICKDataset, self).initialize(word_index, embedding)
        # convert label to 5 probability classes
        new_labels = torch.zeros(self.__len__(), 5)
        for i, sim in enumerate(self.labels):
            ceil, floor = math.ceil(sim), math.floor(sim)
            if ceil == floor:
                new_labels[i][floor - 1] = 1
            else:
                new_labels[i][floor - 1] = ceil - sim
                new_labels[i][ceil - 1] = sim - floor

        self.labels = new_labels


class MSRVIDDataset(MPCNNDataset):

    dataset_root = os.path.join(os.pardir, os.pardir, 'data', 'msrvid')

    def __init__(self, dataset_type):
        super(MSRVIDDataset, self).__init__(dataset_type)

    def initialize(self, word_index, embedding):
        super(MSRVIDDataset, self).initialize(word_index, embedding)
