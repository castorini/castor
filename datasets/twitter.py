import os

import torch

from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from torchtext.data.field import Field
from torchtext.data.iterator import BucketIterator
from torchtext.data.iterator import Iterator
from torchtext.vocab import Vectors
from torchtext.data import Pipeline

from datasets.castor_dataset import CastorPairDataset
from datasets.idf_utils import get_pairwise_word_to_doc_freq, get_pairwise_overlap_features


class TWITTER(Dataset):
    NAME = 'twitter'
    NUM_CLASSES = 2
    ID_FIELD = Field(sequential=False, tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True)
    AID_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    TEXT_FIELD = Field(batch_first=True, tokenize=lambda x: x)  # tokenizer is identity since we already tokenized it to compute external features
    EXT_FEATS_FIELD = Field(tensor_type=torch.FloatTensor, use_vocab=False, batch_first=True, tokenize=lambda x: x,
                            postprocessing=Pipeline(lambda arr, _, train: [float(y) for y in arr]))
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True)
    VOCAB_SIZE = 0

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence_1)

    def __init__(self, pardir, subdirs):
        """
        Create a Twitter dataset instance.
        """
        fields = [('id', self.ID_FIELD), ('sentence_1', self.TEXT_FIELD), ('sentence_2', self.TEXT_FIELD), ('ext_feats',
                self.EXT_FEATS_FIELD), ('label', self.LABEL_FIELD), ('aid', self.AID_FIELD)]

        examples = []
        for subdir in subdirs:
            path = os.path.join(pardir, subdir)
            with open(os.path.join(path, 'a.toks'), 'r') as f1, open(os.path.join(path, 'b.toks'), 'r') as f2:
                sent_list_1 = [l.rstrip('.\n').split(' ') for l in f1]
                sent_list_2 = [l.rstrip('.\n').split(' ') for l in f2]

        word_to_doc_cnt = get_pairwise_word_to_doc_freq(sent_list_1, sent_list_2)
        overlap_feats = get_pairwise_overlap_features(sent_list_1, sent_list_2, word_to_doc_cnt)

        for subdir_i, subdir in enumerate(subdirs):
            path = os.path.join(pardir, subdir)
            with open(os.path.join(path, 'id.txt'), 'r') as id_file, open(os.path.join(path, 'sim.txt'), 'r') as label_file:
                for i, (pair_id, l1, l2, ext_feats, label) in enumerate(zip(id_file, sent_list_1, sent_list_2, overlap_feats, label_file)):
                    pair_id = pair_id.rstrip('.\n')
                    label = label.rstrip('.\n')
                    example_list = [pair_id, l1, l2, ext_feats, label, (subdir_i) * 100000 + (i + 1)]
                    example = Example.fromlist(example_list, fields)
                    examples.append(example)

        super(TWITTER, self).__init__(examples, fields)

    @classmethod
    def splits(cls, path, train_paths, test_paths, **kwargs):
        train_data = cls(path, train_paths, **kwargs)
        test_data = cls(path, test_paths, **kwargs)
        return train_data, test_data

    @classmethod
    def set_vectors(cls, field, vector_path):
        return CastorPairDataset.set_vectors(field, vector_path)

    @classmethod
    def iters(cls, path, train_dirs, test_dirs, vectors_name, vectors_dir, batch_size=64, shuffle=True, device=0, pt_file=False, vectors=None, unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param train_dirs: list of directory names used for training
        :param test_dirs: list of directory name used for testing
        :param vectors_name: name of word vectors file
        :param vectors_dir: directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """

        train, test = cls.splits(path, train_dirs, test_dirs)
        if not pt_file:
            if vectors is None:
                vectors = Vectors(name=vectors_name, cache=vectors_dir, unk_init=unk_init)
            cls.TEXT_FIELD.build_vocab(train, test, vectors=vectors)
        else:
            cls.TEXT_FIELD.build_vocab(train, test)
            cls.TEXT_FIELD = cls.set_vectors(cls.TEXT_FIELD, os.path.join(vectors_dir, vectors_name))

        cls.LABEL_FIELD.build_vocab(train, test)

        cls.VOCAB_SIZE = len(cls.TEXT_FIELD.vocab)

        return BucketIterator.splits((train, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)
