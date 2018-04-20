import importlib
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data

###### CONFIG #########
#######################

import jingweiz

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.circuit = jingweiz.NTMCircuit(args)

    def forward(self, _input, hidden):
        ret = self.circuit(_input)
        return ret, hidden

    def reset_memory(self):
        self.circuit._reset_states()

    def init_hidden(self):
        return None


def get_net(in_vector_len, out_vector_len):
    class Empty():
        #use_cuda = True
        #dtype = torch.cuda.FloatTensor
        use_cuda = False
        dtype = torch.FloatTensor

    args = Empty()
    args.batch_size = 1
    args.input_dim = in_vector_len
    args.output_dim = out_vector_len
    args.hidden_dim = 100
    args.num_write_heads = 1
    args.num_read_heads = 1
    args.mem_hei = 128
    args.mem_wid = 20
    args.clip_value = 20.
    args.controller_params = Empty()
    args.accessor_params = Empty()
    args.accessor_params.write_head_params = Empty()
    args.accessor_params.write_head_params.num_allowed_shifts = 3
    args.accessor_params.read_head_params = Empty()
    args.accessor_params.read_head_params.num_allowed_shifts = 3
    args.accessor_params.memory_params = Empty()
    net = Net(args)
    return net
