from NCE_MP_Pytorch.evaluators.qa_evaluator import QAEvaluator


class WikiQAEvaluator_NCE(QAEvaluator):

    def __init__(self, dataset_cls, model, data_loader, batch_size, device):
        super(WikiQAEvaluator_NCE, self).__init__(dataset_cls, model, data_loader, batch_size, device)
