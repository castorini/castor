import torch
import torch.nn as nn
from torch.autograd import Variable
#from utils import 
import torch.nn.functional as F
from han.sent_level_rnn import SentLevelRNN
from han.word_level_rnn import WordLevelRNN
 

class HAN(nn.Module):
        def __init__(self, config):
                super(HAN, self).__init__()	
                self.dataset = config.dataset
                self.mode = config.mode
                self.word_attention_rnn = WordLevelRNN(config)
                self.sentence_attention_rnn = SentLevelRNN(config)
        def forward(self,x):
                x = x.permute(1,2,0) ## Expected : #sentences, #words, batch size
                num_sentences = x.size()[0]
                word_attentions = None
                for i in range(num_sentences):
                        _word_attention = self.word_attention_rnn(x[i,:,:])
                        if word_attentions is None:
                                word_attentions = _word_attention
                        else:
                                word_attentions = torch.cat((word_attentions, _word_attention),0)
                return self.sentence_attention_rnn(word_attentions)

