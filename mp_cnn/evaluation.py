from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class MPCNNEvaluatorFactory(object):
    """
    Get the corresponding Evaluator class for a particular dataset.
    """
    @staticmethod
    def get_evaluator(dataset_name, model, data_loader):
        if dataset_name == 'sick':
            return SICKEvaluator(model, data_loader)
        elif dataset_name == 'msrvid':
            raise NotImplementedError('msrvid Evaluator is not yet implemented.')
        else:
            raise ValueError('{} is not a valid dataset.'.format(dataset_name))


class Evaluator(object):
    """
    Evaluates performance of model on a Dataset, using metrics specific to the Dataset.
    """

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def get_scores(self):
        """
        Get the scores used to evaluate the model.
        Should return ([score1, score2, ..], [score1_name, score2_name, ...]).
        The first score is the primary score used to determine if the model has improved.
        """
        raise NotImplementedError('Evaluator subclass needs to implement get_score')


class SICKEvaluator(Evaluator):

    def __init__(self, model, data_loader):
        super(SICKEvaluator, self).__init__(model, data_loader)

    def get_scores(self):
        self.model.eval()
        predict_classes = torch.arange(1, 6)
        test_kl_div_loss = 0
        predictions = []
        true_labels = []
        for sentences, labels in self.data_loader:
            sent_a, sent_b = Variable(sentences['a'], volatile=True), Variable(sentences['b'], volatile=True)
            labels = Variable(labels, volatile=True)
            output = self.model(sent_a, sent_b)
            test_kl_div_loss += F.kl_div(output, labels, size_average=False).data[0]
            true_labels.append(predict_classes.dot(labels.data.view(5)))
            predictions.append(predict_classes.dot(output.data.exp()))

        test_kl_div_loss /= len(self.data_loader.dataset)
        pearson_r = pearsonr(predictions, true_labels)[0]
        spearman_r = spearmanr(predictions, true_labels)[0]
        return [pearson_r, spearman_r, test_kl_div_loss], ['pearson_r', 'spearman_r', 'KL-divergence loss']
