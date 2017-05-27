import string
import os
import glob
import nltk
import torch

def build_vocab_SQ(data_dir):
    filepaths = glob.glob(os.path.join(data_dir, 'annotated*.txt'))
    print("reading filepaths: {}".format(filepaths))
    dst_path = os.path.join(data_dir, 'vocab.pt')
    word_vocab = set()
    relation_vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                line_items = line.split("\t")
                qText = line_items[3]
                relation = line_items[1]
                # process text: remove punctuations, lowercase
                punc_remover = str.maketrans('', '', string.punctuation)
                processed_text = qText.lower().translate(punc_remover)
                tokens = nltk.word_tokenize(processed_text)
                word_vocab |= set(tokens)
                relation_vocab |= relation

    wv_dict = {word: i for i, word in enumerate(sorted(word_vocab))}  # word to index dictionary
    relation_dict = {relation: i for i, relation in enumerate(sorted(relation_vocab))}  # relation to index dictionary
    ret = (wv_dict, relation_dict) # tuple to write to dst_path
    print("saving word2index and answer2index dicts to {}".format(dst_path))
    torch.save(ret, dst_path)


print("WARNING: This script is dataset specific. Please change it to fit your own dataset.")
data_dir = '../data/SimpleQuestions_v2/'
print("Building vocab for data in: {}".format(data_dir))
build_vocab_SQ(data_dir)
print("Done!")