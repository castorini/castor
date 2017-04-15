import argparse
import os
import numpy as np
from collections import defaultdict

def read_in_data(data, set, file):
    sentences = []
    with open(os.path.join(data, set, file)) as inf:
        sentences = [line.strip() for line in inf.readlines()]
    return sentences


def compute_idfs(data):
    term_idfs = defaultdict(float)
    for doc in data:
        for term in list(set(doc.split())):
            term_idfs[term] += 1.0
    N = len(data)
    for term, n_t in term_idfs.items():
        term_idfs[term] = np.log(N/1+n_t)
    return term_idfs


def compute_idf_sum_similarity(questions, answers, term_idfs):

    # compute IDF sums for common_terms
    idf_sum_similarity = np.zeros(len(questions))
    for i in range(len(questions)):
        q = questions[i]
        a = answers[i]
        q_terms = set(q.split())
        a_terms = set(a.split())
        common_terms = q_terms.intersection(a_terms)
        idf_sum_similarity[i] = np.sum([term_idfs[term] for term in list(common_terms)])

    return idf_sum_similarity


def write_out_idf_sum_similarities(qids, questions, answers, term_idfs, outfile):
    with open(outfile, 'w') as outf:
        idf_sum_similarity = compute_idf_sum_similarity(questions, answers, term_idfs)
        for i in range(len(questions)):
            print('{} 0 {} 0 {} data_only_idfbaseline'.format(qids[i], i, idf_sum_similarity[i]),
                  file=outf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="uses idf weights from the question-answer pairs only,\
                   and not from the whole corpus")
    ap.add_argument('qa_data', help="path to the QA dataset",
                    choices=['../../data/TrecQA', '../../data/WikiQA'])
    ap.add_argument('outfile_prefix', help="output file prefix")
    ap.add_argument('--ignore-test', help="does not consider test data when computing IDF of terms",
                    action="store_true")

    args = ap.parse_args()

    # read in the data
    train_data, dev_data, test_data = 'train', 'dev', 'test'
    if args.qa_data.endswith('TrecQA'):
        train_data, dev_data, test_data = 'train-all', 'raw-dev', 'raw-test'

    train_que = read_in_data(args.qa_data, train_data, 'a.toks')
    train_ans = read_in_data(args.qa_data, train_data, 'b.toks')

    dev_que = read_in_data(args.qa_data, dev_data, 'a.toks')
    dev_ans = read_in_data(args.qa_data, dev_data, 'b.toks')

    test_que = read_in_data(args.qa_data, test_data, 'a.toks')
    test_ans = read_in_data(args.qa_data, test_data, 'b.toks')

    all_data = train_que + dev_que + train_ans + dev_ans

    if not args.ignore_test:
        all_data += test_ans
        all_data += test_que

    # compute inverse document frequencies for terms
    term_idfs = compute_idfs(all_data)

    # write out in trec_eval format
    write_out_idf_sum_similarities(read_in_data(args.qa_data, train_data, 'id.txt'),
                                   train_que, train_ans, term_idfs,
                                   '{}.{}.idfsim'.format(args.outfile_prefix, train_data))

    write_out_idf_sum_similarities(read_in_data(args.qa_data, dev_data, 'id.txt'),
                                   dev_que, dev_ans, term_idfs,
                                   '{}.{}.idfsim'.format(args.outfile_prefix, dev_data))

    write_out_idf_sum_similarities(read_in_data(args.qa_data, test_data, 'id.txt'),
                                   test_que, test_ans, term_idfs,
                                   '{}.{}.idfsim'.format(args.outfile_prefix, test_data))

