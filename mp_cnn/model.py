import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MPCNN(nn.Module):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths):
        super(MPCNN, self).__init__()

        self.n_word_dim = n_word_dim
        self.n_per_dim_filters = n_per_dim_filters
        self.filter_widths = filter_widths
        self.holistic_conv_layers = {}
        self.per_dim_conv_layers = {}

        for ws in filter_widths:
            self.holistic_conv_layers[ws] = nn.Sequential(
                nn.Conv1d(n_word_dim, n_holistic_filters, ws),
                nn.Tanh()
            )

            self.per_dim_conv_layers[ws] = nn.Sequential(
                nn.Conv1d(n_word_dim, n_word_dim * n_per_dim_filters, ws, groups=n_word_dim),
                nn.Tanh()
            )

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        block_b = {}
        for ws in self.filter_widths:
            holistic_conv_out = self.holistic_conv_layers[ws](sent)
            block_a[ws] = {
                'max': F.max_pool1d(holistic_conv_out, holistic_conv_out.size()[2]),
                'min': F.max_pool1d(-1 * holistic_conv_out, holistic_conv_out.size()[2]),
                'mean': F.avg_pool1d(holistic_conv_out, holistic_conv_out.size()[2])
            }

            per_dim_conv_out = self.per_dim_conv_layers[ws](sent)
            per_dim_conv_out = per_dim_conv_out.view(self.n_word_dim, self.n_per_dim_filters, -1)
            block_b[ws] = {
                'max': F.max_pool2d(per_dim_conv_out, (1, per_dim_conv_out.size()[2])),
                'min': F.max_pool2d(-1 * per_dim_conv_out, (1, per_dim_conv_out.size()[2]))
            }
        return block_a, block_b

    def forward(self, sent1, sent2):
        sent1_block_a, sent1_block_b = self._get_blocks_for_sentence(sent1)
        sent2_block_a, sent2_block_b = self._get_blocks_for_sentence(sent2)

        # TODO handle ws = infinity
        # TODO implement similarity measurement layer and fully-connected layer
        # return dummy return values for now
        return Variable(torch.Tensor([0, 0]))
