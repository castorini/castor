import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class BiLSTM1(nn.Module):

    def __init__(self, config):
        super(BiLSTM1, self).__init__()
        self.config = config

        self.lstm = nn.LSTM(input_size=config.d_embedding, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=config.dropout_prob,
                        bidirectional=False)

        ## FIXME: BiLSTM has more cells or sth
        # self.lstm = nn.LSTM(input_size=config.d_embedding, hidden_size=config.d_hidden,
        #                 num_layers=config.n_layers, dropout=config.dropout_prob,
        #                 bidirectional=config.birnn)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2rel = nn.Linear(config.d_hidden, config.d_out)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.config.n_layers, 1, self.config.d_hidden)),
                        Variable(torch.zeros(self.config.n_layers, 1, self.config.d_hidden)))

    # embeds is Variable of size - (|S|, |D|)
    def forward(self, embeds):
        batch_size = 1 # fixed to 1
        sentence_length = embeds.data.size()[0]
        lstm_out, self.hidden = self.lstm(embeds.view(sentence_length, batch_size, -1), self.hidden)
        ht = self.hidden[0] # hidden = (ht, ct)
        # rel_space = self.hidden2rel(lstm_out.view(sentence_length, -1))
        rel_space = self.hidden2rel(ht[-1].view(batch_size, -1)) # size - (1, |K|)
        rel_scores = F.log_softmax(rel_space)
        return rel_scores