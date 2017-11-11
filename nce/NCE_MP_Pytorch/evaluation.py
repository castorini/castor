from nce.NCE_MP_Pytorch.evaluators.sick_evaluator import SICKEvaluator
from nce.NCE_MP_Pytorch.evaluators.msrvid_evaluator import MSRVIDEvaluator
from nce.NCE_MP_Pytorch.evaluators.trecqa_evaluator import TRECQAEvaluator
from nce.NCE_MP_Pytorch.evaluators.wikiqa_evaluator import WikiQAEvaluator


class MPCNNEvaluatorFactory(object):
    """
    Get the corresponding Evaluator class for a particular dataset.
    """
    evaluator_map = {
        'sick': SICKEvaluator,
        'msrvid': MSRVIDEvaluator,
        'trecqa': TRECQAEvaluator,
        'wikiqa': WikiQAEvaluator
    }

    @staticmethod
    def get_evaluator(dataset_cls, model, data_loader, batch_size, device):
        if data_loader is None:
            return None

        if not hasattr(dataset_cls, 'NAME'):
            raise ValueError('Invalid dataset. Dataset should have NAME attribute.')

        if dataset_cls.NAME not in MPCNNEvaluatorFactory.evaluator_map:
            raise ValueError('{} is not implemented.'.format(dataset_cls))

        return MPCNNEvaluatorFactory.evaluator_map[dataset_cls.NAME](
            dataset_cls, model, data_loader, batch_size, device
        )
