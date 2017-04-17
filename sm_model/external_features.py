# module to compute various external features for the sm_model.
# TODO: add more external features like:
# word mover distance, cosine sim in tf.idf space, cosine sim in word embedding space
# overlap based on parts of speech: noun, verb, adj (POS tag)
# word embedding cosine sim based on part of speech: noun, verb, adj

import string
from collections import defaultdict

import numpy as np

import nltk
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def stopped(sentences):
    """
    remove stop words from given sentences (questions|answers)
    """
    stoplist = set(stopwords.words('english'))
    stoplist.update(set(string.punctuation))
    def stop(sentence):
        return ' '.join([word for word in sentence.split() if word not in stoplist])
    return [stop(sentence) for sentence in sentences]

def stemmed(sentences):
    """
    reduce sentence terms to stemmed representations
    """
    stemmer = PorterStemmer()
    def stem(sentence):
        return ' '.join([stemmer.stem(word) for word in sentence.split()])
    return [stem(sentence) for sentence in sentences]

def get_qadata_only_idf(all_data):
    """
    returns idf weights computed over all question answer pairs in the dataset
    """
    if not type(all_data) is set:
        all_data = set(all_data)
    term_idfs = defaultdict(float)
    for doc in list(all_data):
        for term in list(set(doc.split())):
            term_idfs[term] += 1.0
    N = len(all_data)
    for term, n_t in term_idfs.items():
        term_idfs[term] = np.log(N/(1+n_t))
    return term_idfs

def get_source_corpus_idf(all_data):
    """
    fetches idf weights from source corpus (disks1-5+aquaint|wikipedia) index, for all the qa pairs
    """
    pass

def compute_overlap(questions, answers):
    """
    returns simple overlap between document pairs
    """
    overlap_scores = []
    for q, a in zip(questions, answers):
        q_terms = set(q.split())
        a_terms = set(a.split())
        common_terms = q_terms.intersection(a_terms)
        overlap = float(len(common_terms))/(len(q_terms) + len(a_terms))
        overlap_scores.append(overlap)
    return np.array(overlap_scores)

def compute_idf_weighted_overlap(questions, answers, idf_weights):
    """
    returns idf weighted overlap
    """
    overlap_scores = []
    for q, a in zip(questions, answers):
        q_terms = set(q.split())
        a_terms = set(a.split())
        common_terms = q_terms.intersection(a_terms)
        idf_weighted_overlap = np.sum([idf_weights[term] for term in list(common_terms)])
        idf_weighted_overlap /= (len(q_terms) + len(a_terms))
        overlap_scores.append(idf_weighted_overlap)
    return np.array(overlap_scores)


def set_external_features_as_per_paper(trainer):
    """
    computes external features as per the paper AND saves them into trainer
    """
    all_questions, all_answers = [], []
    for split in trainer.data_splits.keys():
        questions, answers, labels, max_q_len, max_a_len, default_ext_feats = \
            trainer.data_splits[split]
        all_questions.extend(questions)
        all_answers.extend(answers)

    all_data = set(all_questions + all_answers)
    idf_weights = get_qadata_only_idf(all_data)

    external_features = {}

    for split in trainer.data_splits.keys():
        questions, answers, labels, max_q_len, max_a_len, default_ext_feats = \
            trainer.data_splits[split]

        overlap = compute_overlap(questions, answers)
        idf_weighted_overlap = compute_idf_weighted_overlap(questions, answers, idf_weights)
        overlap_no_stopwords =\
            compute_overlap(stopped(questions), stopped(answers))
        idf_weighted_overlap_no_stopwords =\
            compute_idf_weighted_overlap(stopped(questions), stopped(answers), idf_weights)
        ext_feats = [np.array(feats) for feats in zip(overlap, idf_weighted_overlap,\
                    overlap_no_stopwords, idf_weighted_overlap_no_stopwords)]
        trainer.data_splits[split][-1] = ext_feats
        external_features[split] = ext_feats
    return external_features
