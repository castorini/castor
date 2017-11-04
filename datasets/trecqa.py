import os

import torch
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from torchtext.data.field import Field
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from datasets.idf_utils import get_pairwise_word_to_doc_freq, get_pairwise_overlap_features


class TRECQA(Dataset):
    NAME = 'trecqa'
    NUM_CLASSES = 2
    ID_FIELD = Field(sequential=False, tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True)
    TEXT_FIELD = Field(batch_first=True, tokenize=lambda x: x)  # tokenizer is identity since we already tokenized it to compute external features
    EXT_FEATS_FIELD = Field(tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True, tokenize=lambda x: x)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence_1)

    def __init__(self, path):
        """
        Create a TRECQA dataset instance
        """
        fields = [('id', self.ID_FIELD), ('sentence_1', self.TEXT_FIELD), ('sentence_2', self.TEXT_FIELD), ('ext_feats', self.EXT_FEATS_FIELD), ('label', self.LABEL_FIELD)]

        examples = []
        with open(os.path.join(path, 'a.toks'), 'r') as f1, open(os.path.join(path, 'b.toks'), 'r') as f2:
            sent_list_1 = [l.rstrip('.\n').split(' ') for l in f1]
            sent_list_2 = [l.rstrip('.\n').split(' ') for l in f2]

        word_to_doc_cnt = get_pairwise_word_to_doc_freq(sent_list_1, sent_list_2)
        overlap_feats = get_pairwise_overlap_features(sent_list_1, sent_list_2, word_to_doc_cnt)

        with open(os.path.join(path, 'id.txt'), 'r') as id_file, open(os.path.join(path, 'sim.txt'), 'r') as label_file:
            for pair_id, l1, l2, ext_feats, label in zip(id_file, sent_list_1, sent_list_2, overlap_feats, label_file):
                pair_id = pair_id.rstrip('.\n')
                label = label.rstrip('.\n')
                example = Example.fromlist([pair_id, l1, l2, ext_feats, label], fields)
                examples.append(example)

        super(TRECQA, self).__init__(examples, fields)

    @classmethod
    def splits(cls, path, train='train-all', validation='raw-dev', test='raw-test', **kwargs):
        return super(TRECQA, cls).splits(path, train=train, validation=validation, test=test, **kwargs)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None, unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, validation, test = cls.splits(path)

        cls.TEXT_FIELD.build_vocab(train, validation, test, vectors=vectors)

        return BucketIterator.splits((train, validation, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)
