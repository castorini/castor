import torch
import torch.nn as nn

import torch.nn.functional as F

class SmPlusPlus(nn.Module):
    def __init__(self, config):
        super(SmPlusPlus, self).__init__()
        output_channel = config.output_channel
        target_class = config.target_class
        questions_num = config.questions_num
        answers_num = config.answers_num
        words_dim = config.words_dim
        filter_width = config.filter_width
        self.mode = config.mode
        self.use_ext = config.use_ext

        print(questions_num, words_dim)
        n_classes = 2
        ext_feats_size = 4

        if self.mode == 'multichannel':
            input_channel = 2
        else:
            input_channel = 1

        self.static_question_embed = nn.Embedding(questions_num, words_dim)
        self.nonstatic_question_embed = nn.Embedding(questions_num, words_dim)
        self.static_answer_embed = nn.Embedding(answers_num, words_dim)
        self.nonstatic_answer_embed = nn.Embedding(answers_num, words_dim)
        self.nonstatic_question_embed.weight.requires_grad = False
        self.nonstatic_answer_embed.weight.requires_grad = False

        self.conv_q = nn.Conv2d(input_channel, output_channel, (filter_width, words_dim), padding=(filter_width - 1, 0))
        self.conv_a = nn.Conv2d(input_channel, output_channel, (filter_width, words_dim), padding=(filter_width - 1, 0))

        self.dropout = nn.Dropout(config.dropout)
        # n_hidden = 2 * output_channel + (0 if self.use_ext else ext_feats_size)

        n_hidden = 2 * output_channel
        self.hidden = nn.Linear(n_hidden, n_classes)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x_question = x.question
        x_answer = x.answer
        x_ext = x.ext_feat

        # this mode need not be tried
        if self.mode == 'rand':
            question = self.embed(x_question)
            answer = self.embed(x_answer) # (batch, sent_len, embed_dim)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        # this is the actual SM model mode
        elif self.mode == 'static':
            question = self.static_question_embed(x_question)
            answer = self.static_answer_embed(x_answer) # (batch, sent_len, embed_dim)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        elif self.mode == 'non-static':
            question = self.nonstatic_question_embed(x_question)
            answer = self.nonstatic_answer_embed(x_answer) # (batch, sent_len, embed_dim)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        elif self.mode == 'multichannel':
            question_static = self.static_question_embed(x_question)
            answer_static = self.static_answer_embed(x_answer)
            question_nonstatic = self.nonstatic_question_embed(x_question)
            answer_nonstatic = self.nonstatic_answer_embed(x_answer)
            question = torch.stack([question_static, question_nonstatic], dim=1)
            answer = torch.stack([answer_static, answer_nonstatic], dim=1)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
        else:
            print("Unsupported Mode")
            exit()

        x = torch.cat(x, 1)
        # if x_ext:
        #     x = torch.cat([x, x_ext], 1)
        x = self.dropout(x)
        x = self.hidden(x)
        logit = self.logsoftmax(x)

        return logit