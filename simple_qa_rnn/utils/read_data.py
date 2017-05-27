from .vocab import Vocab
import torch

## functions for loading data from disk

def read_embedding(embed_pt_filepath):
    embed_tuple = torch.load(embed_pt_filepath)
    word2index, w2v_tensor, dim = embed_tuple
    return word2index, w2v_tensor

def read_text(text, word_vocab):
    out_text = []
    for tokens in text:
        S = len(tokens)
        sent = torch.IntTensor(max(S, 3))
        for i in range(S):
            token = tokens[i]
            sent[i] = word_vocab.index(token)
        if S < 3:
            for i in range(S, 3):
                sent[i] = word_vocab.unk_index
        out_text.append(sent)
    return out_text

def read_labels(rel_labels, rel_vocab):
    N = len(rel_labels)
    label_tensor = torch.IntTensor(N)
    for i in range(N):
        token = rel_labels[i]
        label_tensor[i] = rel_vocab.index(token)
    return label_tensor

def read_dataset(dataset_pt_filepath):
    dataset_tuple = torch.load(dataset_pt_filepath)
    w2i_dict, rel2i_dict, questions, rel_labels = dataset_tuple

    word_vocab = Vocab(w2i_dict)
    word_vocab.add_unk_token("<UNK>")
    rel_vocab = Vocab(rel2i_dict)
    questions_tensor = read_text(questions, word_vocab)
    rel_labels_tensor = read_labels(rel_labels, rel_vocab)

    dataset = {"word_vocab": word_vocab, "rel_vocab": rel_vocab,
                    "questions": questions_tensor, "rel_labels": rel_labels_tensor}
    return dataset
