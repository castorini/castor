import string
import os
import glob
import nltk
import torch

def build_vocab_SQ(data_dir):
    filepaths = glob.glob(os.path.join(data_dir, 'annotated*.txt'))
    print("reading filepaths: {}".format(filepaths))
    word_vocab = set()
    relation_vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                line_items = line.split("\t")
                # add relation
                relation = line_items[1]
                relation_vocab.add(relation)
                # process/add text: remove punctuations, lowercase
                qText = line_items[3]
                punc_remover = str.maketrans('', '', string.punctuation)
                processed_text = qText.lower().translate(punc_remover)
                tokens = nltk.word_tokenize(processed_text)
                word_vocab |= set(tokens)

    wv_dict = {word: i for i, word in enumerate(sorted(word_vocab))}  # word to index dictionary
    relation_dict = {relation: i for i, relation in enumerate(sorted(relation_vocab))}  # relation to index dictionary
    return (wv_dict, relation_dict)


print("WARNING: This script is dataset specific. Please change it to fit your own dataset.")
data_dir = 'data/SimpleQuestions_v2/'
dst_path = os.path.join(data_dir, 'vocab.pt')
print("Building vocab for data in: {}".format(data_dir))
ret = build_vocab_SQ(data_dir) # tuple to write to dst_path
print("saving word2index and answer2index dicts to {}".format(dst_path))
torch.save(ret, dst_path)
print("Done!")