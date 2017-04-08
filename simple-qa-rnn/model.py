import torch
import torch.nn as nn
from torch.autograd import Variable

class BiLSTM1(nn.Module):
    def __init__(self, config):
        super(BiLSTM1, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=config.d_embedding, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=config.dropout_ratio,
                        bidirectional=config.birnn)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, 1, self.config.d_hidden)),
                        Variable(torch.zeros(1, 1, self.config.d_hidden)))

    def forward(self, inputs):
        pass