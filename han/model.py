import torch
import torch.nn as nn
from torch.autograd import Variable
#from utils import 
import torch.nn.functional as F

class AttentionRNN(nn.Module):
	def __init__(self, config):
		super(AttentionRNN, self).__init__()	
		dataset = config.dataset
		print("Hey JEu")
		print(config)
		word_num_hidden = config.word_num_hidden
		words_num = config.words_num
		words_dim = config.words_dim
		self.mode = config.mode
		if config.mode == 'rand':
			rand_embed_init = torch.Tensor(words_num, words_dim).uniform_(-0.25, 0.25)
			self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
		elif config.mode == 'static':
			self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
		elif config.mode == 'non-static':
			self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
		else:
			print("Unsuported order")
			exit()
		
		self.word_context_wghts = nn.Parameter(torch.rand(2*word_num_hidden,1))
		self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional = True)
		self.linear = nn.Linear(2*word_num_hidden, 2*word_num_hidden, bias = True) ## 
		self.word_context_wghts.data.uniform_(-0.1, 0.1)
                
		sentence_num_hidden = config.sentence_num_hidden
		target_class = config.target_class
		self.sentence_context_wghts = nn.Parameter(torch.rand(2*sentence_num_hidden, 1))
		self.sentence_context_wghts.data.uniform_(-0.1,0.1)
		self.sentence_GRU = nn.GRU(2*word_num_hidden , sentence_num_hidden, bidirectional = True)
		self.sentence_linear = nn.Linear(2*sentence_num_hidden, 2*sentence_num_hidden, bias = True)
		self.fc = nn.Linear(2*sentence_num_hidden, target_class)
		self.soft_word = nn.Softmax()
		self.soft_sent = nn.Softmax()
		self.final_soft = nn.Softmax()
	def forward(self,x):
		print(x.shape)
		if self.mode == 'rand':
			x = self.embed(x)
		elif self.mode == 'static':
			x = self.static_embed(x)
		elif self.mode == 'non-static':
			x = self.non_static_embed(x)
		else :
			print("Unsuported mode")
			exit()
		x = x.permute(1,0,2)
		print(x.shape)
		h,_ = self.GRU(x)
		x = torch.tanh(self.linear(h))
		x = torch.matmul(x, self.word_context_wghts)
		x = x.squeeze()
		x = self.soft_word(x.transpose(1,0))
		x = torch.mul(h.permute(2,0,1), x.transpose(1,0))
		x = torch.sum(x, dim = 1).transpose(1,0).unsqueeze(0)

		sentence_h,_ = self.sentence_GRU(x)
		x = torch.tanh(self.sentence_linear(sentence_h))
		x = torch.matmul(x, self.sentence_context_wghts)
		x = x.squeeze()
		print(x.shape)
		x = self.soft_sent(x)
		print(x.shape)
		print("hi")
		print(h.shape)
		x = torch.mul(h.permute(2,0,1), x)
		x = torch.sum(x,dim = 1).transpose(1,0).unsqueeze(0)
		x = self.final_soft(self.fc(x.squeeze(0)))
		return x	
"""
class SentenceAttentionModel(nn.Module):
	def __init__(self, config):
		dataset = config.dataset
		sentence_num_hidden = config.sentence_num_hidden
		num_classes = config.num_classes
		target_class = config.target_class
		word_num_hidden = config.word_num_hidden
		
		self.sentence_context_wghts = nn.Parameters(torch.rand(2*num_sentence_hidden,1))
		self.GRU = nn.GRU(2*word_num_hidden, sentence_num_hidden)
		self.linear = nn.Linear(2*sentence_num_hidden,2*sentence_num_hidden,bias = True)
		self.sentence_context_wghts.data.uniform_(-0.1,0.1)
		self.fc = nn.Linear(2*sentence_num_hidden, target_class)
		
	def forward(self,x):
		h = self.GRU(x)
		x = torch.tanh(self.linear(h))
		x = torch.matmul(x,self.sentence_context_wghts)
		x = x.squeeze()
		x = nn.Softmax(x.transpose(1,0))
		x = torch.mul(h.permute(2,0,1), x.tranpose(1,0))
		x = torch.sum(x,dim = 1).transpose(1,0).unsqueeze(0)
		x = nn.log_softmax(self.fc(x.squeeze(0)))
		return x
"""


