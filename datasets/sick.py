import math
import os

import numpy as np
import torch
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from torchtext.data.field import Field
from torchtext.data.iterator import BucketIterator
from torchtext.data.pipeline import Pipeline
from torchtext.vocab import Vectors

from datasets.idf_utils import get_pairwise_word_to_doc_freq, get_pairwise_overlap_features


def get_class_probs(sim, *args):
    """
    Convert a single label into class probabilities.
    """
    class_probs = np.zeros(SICK.NUM_CLASSES)
    ceil, floor = math.ceil(sim), math.floor(sim)
    if ceil == floor:
        class_probs[floor - 1] = 1
    else:
        class_probs[floor - 1] = ceil - sim
        class_probs[ceil - 1] = sim - floor

    return class_probs


class SICK(Dataset):
    NAME = 'sick'
    NUM_CLASSES = 5
    ID_FIELD = Field(sequential=False, tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True)
    TEXT_FIELD = Field(batch_first=True, tokenize=lambda x: x)  # tokenizer is identity since we already tokenized it to compute external features
    EXT_FEATS_FIELD = Field(tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True, tokenize=lambda x: x)
    LABEL_FIELD = Field(sequential=False, tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True, postprocessing=Pipeline(get_class_probs))

    @staticmethod
    def sort_key(ex):
        return len(ex.a)

    def __init__(self, path):
        """
        Create a SICK dataset instance
        """
        fields = [('id', self.ID_FIELD), ('a', self.TEXT_FIELD), ('b', self.TEXT_FIELD), ('ext_feats', self.EXT_FEATS_FIELD), ('label', self.LABEL_FIELD)]

        examples = []
        f1 = open(os.path.join(path, 'a.txt'), 'r')
        f2 = open(os.path.join(path, 'b.txt'), 'r')
        id_file = open(os.path.join(path, 'id.txt'), 'r')
        label_file = open(os.path.join(path, 'sim.txt'), 'r')

        sent_list_1 = [l.rstrip('.\n').split(' ') for l in f1]
        sent_list_2 = [l.rstrip('.\n').split(' ') for l in f2]

        word_to_doc_cnt = get_pairwise_word_to_doc_freq(sent_list_1, sent_list_2)
        overlap_feats = get_pairwise_overlap_features(sent_list_1, sent_list_2, word_to_doc_cnt)

        sent_list_2_iter, overlap_feats_iter = iter(sent_list_2), iter(overlap_feats)
        for l1 in sent_list_1:
            l2 = next(sent_list_2_iter)
            label = label_file.readline().rstrip('.\n')
            pair_id = id_file.readline().rstrip('.\n')
            ext_feats = next(overlap_feats_iter)
            example = Example.fromlist([pair_id, l1, l2, ext_feats, label], fields)
            examples.append(example)

        map(lambda f: f.close(), [f1, f2, label_file])

        super(SICK, self).__init__(examples, fields)

    @classmethod
    def splits(cls, path, train='train', validation='dev', test='test', **kwargs):
        return super(SICK, cls).splits(path, train=train, validation=validation, test=test, **kwargs)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None, unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path)

        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)

        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)
