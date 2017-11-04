import os
import subprocess
import time

import torch.nn.functional as F

from mp_cnn.evaluators.evaluator import Evaluator


def get_map_mrr(qids, predictions, labels, device):
    qrel_fname = 'trecqa_{}_{}.qrel'.format(time.time(), device)
    results_fname = 'trecqa_{}_{}.results'.format(time.time(), device)
    qrel_template = '{qid} 0 {docno} {rel}\n'
    results_template = '{qid} 0 {docno} 0 {sim} mpcnn\n'
    with open(qrel_fname, 'w') as f1, open(results_fname, 'w') as f2:
        docnos = range(len(qids))
        for qid, docno, predicted, actual in zip(qids, docnos, predictions, labels):
            f1.write(qrel_template.format(qid=qid, docno=docno, rel=actual))
            f2.write(results_template.format(qid=qid, docno=docno, sim=predicted))

    trec_out = subprocess.check_output(['../utils/trec_eval-9.0.5/trec_eval', '-m', 'map', '-m', 'recip_rank', qrel_fname, results_fname])
    trec_out_lines = str(trec_out, 'utf-8').split('\n')
    mean_average_precision = float(trec_out_lines[0].split('\t')[-1])
    mean_reciprocal_rank = float(trec_out_lines[1].split('\t')[-1])

    os.remove(qrel_fname)
    os.remove(results_fname)

    return mean_average_precision, mean_reciprocal_rank


class TRECQAEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        super(TRECQAEvaluator, self).__init__(dataset_cls, model, data_loader, batch_size, device)

    def get_scores(self):
        self.model.eval()
        test_cross_entropy_loss = 0
        qids = []
        true_labels = []
        predictions = []

        for batch in self.data_loader:
            qids.extend(batch.id.data.cpu().numpy())
            output = self.model(batch.a, batch.b, batch.ext_feats)
            test_cross_entropy_loss += F.cross_entropy(output, batch.label, size_average=False).data[0]

            true_labels.extend(batch.label.data.cpu().numpy())
            predictions.extend(output.data.exp()[:, 1].cpu().numpy())

            del output

        qids = list(map(lambda n: int(round(n * 10, 0)) / 10, qids))

        mean_average_precision, mean_reciprocal_rank = get_map_mrr(qids, predictions, true_labels, self.data_loader.device)
        test_cross_entropy_loss /= len(batch.dataset.examples)

        return [test_cross_entropy_loss, mean_average_precision, mean_reciprocal_rank], ['cross entropy loss', 'map', 'mrr']
