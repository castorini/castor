from .vocab import Vocab
import torch
import nltk
import string

## functions for loading data from disk

def process_tokenize_text(text):
    punc_remover = str.maketrans('', '', string.punctuation)
    processed_text = text.lower().translate(punc_remover)
    tokens = nltk.word_tokenize(processed_text)
    return tokens

def read_embedding(embed_pt_filepath):
    embed_tuple = torch.load(embed_pt_filepath)
    word2index, w2v_tensor, dim = embed_tuple
    return word2index, w2v_tensor

def read_text(text, word_vocab):
    out_text = []
    for tokens in text:
        S = len(tokens)
        sent = []
        for i in range(S):
            token = tokens[i]
            sent.append( word_vocab.get_index(token) )
        if S < 3:
            for i in range(S, 3):
                sent.append( word_vocab.unk_index )
        out_text.append(sent)
    return torch.LongTensor(out_text)

def read_labels(rel_labels, rel_vocab):
    N = len(rel_labels)
    label_tensor = torch.IntTensor(N)
    for i in range(N):
        token = rel_labels[i]
        label_tensor[i] = rel_vocab.get_index(token)
    return label_tensor

def read_dataset(datapath, word_vocab, rel_vocab):
    questions = []
    rel_labels = []
    # read questions and label from the datapath - could be train, dev, testls
    with open(datapath) as f:
        for line in f:
            line_items = line.split("\t")
            # add relation
            relation = line_items[1]
            rel_labels.append(relation)
            # add text
            qText = line_items[3]
            tokens = process_tokenize_text(qText)
            questions.append(tokens)

    questions_tensor = read_text(questions, word_vocab)
    rel_labels_tensor = read_labels(rel_labels, rel_vocab)

    dataset = {"word_vocab": word_vocab, "rel_vocab": rel_vocab, "size": len(rel_labels),
                                "questions": questions_tensor, "rel_labels": rel_labels_tensor}
    return dataset

