import json
import os
import sys
from collections import Counter
import argparse
import random

import numpy as np
import torch
from nltk.tokenize import TreebankWordTokenizer
from torchtext import  data

from sm_cnn import model
from sm_cnn.external_features import compute_overlap, compute_idf_weighted_overlap, stopped
from trec_dataset import TrecDataset
from wiki_dataset import WikiDataset
from anserini_dependency.RetrieveSentences import RetrieveSentences

sys.modules['model'] = model

class SMModelBridge(object):

    def __init__(self, args):
        if not args.cuda:
            args.gpu = -1
        if torch.cuda.is_available() and args.cuda:
            print("Note: You are using GPU for training")
            torch.cuda.set_device(args.gpu)
            torch.cuda.manual_seed(args.seed)
        if torch.cuda.is_available() and not args.cuda:
            print("Warning: You have Cuda but do not use it. You are using CPU for training")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.QID = data.Field(sequential=False)
        self.QUESTION = data.Field(batch_first=True)
        self.ANSWER = data.Field(batch_first=True)
        self.LABEL = data.Field(sequential=False)
        self.EXTERNAL = data.Field(sequential=True, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False,
                              postprocessing=data.Pipeline(lambda arr, _, train: [float(y) for y in arr]))


        if 'TrecQA' in args.dataset:
            train, dev, test = TrecDataset.splits(self.QID, self.QUESTION, self.ANSWER, self.EXTERNAL, self.LABEL)
        elif 'WikiQA' in args.dataset:
            train, dev, test = WikiDataset.splits(self.QID, self.QUESTION, self.ANSWER, self.EXTERNAL, self.LABEL)
        else:
            print("Unsupported dataset")
            exit()

        self.QID.build_vocab(train, dev, test)
        self.QUESTION.build_vocab(train, dev, test)
        self.ANSWER.build_vocab(train, dev, test)
        self.LABEL.build_vocab(train, dev, test)
        self.retrieveSentencesObj = RetrieveSentences(args)
        self.idf_json = self.retrieveSentences.getTermIdfJSON()

    def parse(self, sentence):
        s_toks = TreebankWordTokenizer().tokenize(sentence)
        sentence = ' '.join(s_toks).lower()
        return sentence

    def rerank_candidate_answers(self, question, answers):
        # run through the model
        scores_sentences = []
        question = self.parse(question)
        term_idfs = json.loads(self.idf_json)
        term_idfs = dict((k, float(v)) for k, v in term_idfs.items())

        for term in question.split():
            if term not in term_idfs:
                term_idfs[term] = 0.0

        for answer in answers:
            answer = self.parse(answer)
            for term in answer.split():
                if term not in term_idfs:
                    term_idfs[term] = 0.0
    
            overlap = compute_overlap([question], [answer])
            idf_weighted_overlap = compute_idf_weighted_overlap([question], [answer], term_idfs)
            overlap_no_stopwords =\
                compute_overlap(stopped([question]), stopped([answer]))
            idf_weighted_overlap_no_stopwords =\
                compute_idf_weighted_overlap(stopped([question]), stopped([answer]), term_idfs)
            ext_feats = [np.array(feats) for feats in zip(overlap, idf_weighted_overlap,\
                        overlap_no_stopwords, idf_weighted_overlap_no_stopwords)]

            fields = [('question', self.QUESTION), ('answer', self.ANSWER), ('ext_feat', self.EXTERNAL)]
            example = data.Example.fromList([question, answer, ext_feats], fields)
            self.QUESTION.numericalize(self.QUESTION.pad([example.question]))
            self.ANSWER.numericalize(self.ANSWER.pad([example.answer]))
            self.EXTERNAL.numericalize(self.EXTERNAL.pad([example.ext_feat]))
            model.eval()
            scores = model(self.QUESTION, self.ANSWER, self.EXTERNAL)
            print(scores[:, 2].cpu().data.numpy())
            scores_sentences.append((scores[:, 2].cpu().data.numpy(), answer))

        return scores_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bridge Demo. Produces scores in trec_eval format",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help="the path to the saved model file")
    parser.add_argument('--dataset', help="the QA dataset folder {TrecQA|WikiQA}", default='../../data/TrecQA/')
    parser.add_argument("--index", help="Lucene index", required=True)
    parser.add_argument("--embeddings", help="Path of the word2vec index", default="")
    parser.add_argument("--topics", help="topics file", default="")
    parser.add_argument("--query", help="a single query", default="where was newton born ?")
    parser.add_argument("--hits", help="max number of hits to return", default=100)
    parser.add_argument("--scorer", help="passage scores", default="Idf")
    parser.add_argument("--k", help="top-k passages to be retrieved", default=1)

    args = parser.parse_args()

    smmodel = SMModelBridge()

    train_set, dev_set, test_set = 'train', 'dev', 'test'
    if 'TrecQA' in args.dataset_folder:
        train_set, dev_set, test_set = 'train-all', 'raw-dev', 'raw-test'

    for split in [dev_set, test_set]:
        outfile = open('bridge.{}.scores'.format(split), 'w')

        questions = [q.strip() for q in open(os.path.join(args.dataset_folder, split, 'a.toks')).readlines()]
        answers = [q.strip() for q in open(os.path.join(args.dataset_folder, split, 'b.toks')).readlines()]
        labels = [q.strip() for q in open(os.path.join(args.dataset_folder, split, 'sim.txt')).readlines()]
        qids = [q.strip() for q in open(os.path.join(args.dataset_folder, split, 'id.txt')).readlines()]

        qid_question = dict(zip(qids, questions))
        q_counts = Counter(questions)

        answers_offset = 0
        docid_counter = 0

        all_questions_answers = questions + answers
        idf_json = SMModelBridge.get_term_idf_json_list(all_questions_answers)

        for qid, question in sorted(qid_question.items(), key=lambda x: float(x[0])):
            num_answers = q_counts[question]
            q_answers = answers[answers_offset: answers_offset + num_answers]
            answers_offset += num_answers
            sentence_scores = smmodel.rerank_candidate_answers(question, q_answers)

            for score, sentence in sentence_scores:
                print('{} Q0 {} 0 {} sm_cnn_bridge.{}.run'.format(
                    qid,
                    docid_counter,
                    score,
                    os.path.basename(args.dataset_folder)
                ), file=outfile)
                docid_counter += 1
            if 'WikiQA' in args.dataset_folder:
                docid_counter = 0

        outfile.close()
