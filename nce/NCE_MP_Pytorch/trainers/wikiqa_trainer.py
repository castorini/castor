from nce.NCE_MP_Pytorch.trainers.qa_trainer import QATrainer


class WikiQATrainer_NCE(QATrainer):

    def __init__(self, model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None):
        super(WikiQATrainer_NCE, self).__init__(model, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
