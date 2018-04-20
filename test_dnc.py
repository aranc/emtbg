import importlib
import sys
import time
import os.path
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
from utils import notify


###### CONFIG #########
learning_rate = 1e-4
save_every = 60
verbose_every = 1
notify_every = 60 * 30
#######################
clip_grad = 50


def usage():
    print("python3 test_dnc.py challenge net weights.pkl")


def print_matrix(mat):
    #print(mat)
    #return
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0.5:
                print("1", end="")
            else:
                print("0", end="")
        print()


def main(challenge_module_name, net_module_name, weights_filename):
    challenge = importlib.import_module(challenge_module_name)
    challenge = challenge.get_challenge()
    
    in_vector_len = challenge.get_in_vector_len()
    out_vector_len = challenge.get_out_vector_len()

    net = importlib.import_module(net_module_name)
    net = net.get_net(in_vector_len, out_vector_len)
    
    if os.path.isfile(weights_filename):
        print("Loading from file..")
        state_dict = torch.load(weights_filename)
        net.load_state_dict(state_dict)

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, momentum=0.9, alpha=0.9, eps=1e-10)

    last_save = time.time()
    last_notify = time.time()
    last_verbose = 0
    counter = 0
    last_counter = 0

    while True:
        counter += 1
        net.reset_memory()
        hidden = net.init_hidden()
        empty = Variable(torch.Tensor(np.zeros(challenge.get_in_vector_len())))
        _input, target = challenge.get_next_task()
        for i in range(_input.shape[1]):
            _, hidden = net(Variable(torch.Tensor(_input[:,i])), hidden)
        output = []
        for i in range(target.shape[1]):
            _output, hidden = net(empty, hidden)
            output.append(_output.view((-1,1)))

        output = torch.cat(output, 1)
        loss = F.binary_cross_entropy(output, Variable(torch.Tensor(target)))

        if verbose_every is not None:
            if time.time() - last_verbose > verbose_every:
                print("Input:")
                print_matrix(_input)
                print("Target:")
                print_matrix(target)
                print("Output:")
                print_matrix(output.data.numpy())
                _output = output.data.numpy().copy()
                _output[_output > 0.5] = 1
                _output[_output < 0.5] = 0
                print(output.data.numpy())
                print(_input[-1,-1],net_module_name, "loss:", loss.data.numpy()[0])
                _sum = sum(_output != target)
                print("bad bits:", sum(_sum), _sum)
                print("counter:", counter, "("+str(counter-last_counter)+")")
                last_counter = counter
                last_verbose = time.time()

        if notify_every is not None:
            if time.time() - last_notify > notify_every:
                notify(challenge_module_name + " " + net_module_name + " " + str(counter) + " " + str(loss.data.numpy()[0]))
                last_notify = time.time()

        #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        #print([np.sum(_.data.numpy()) for _ in net.parameters()])
        #print(sum([np.sum(_.data.numpy()) for _ in net.parameters()]))
        loss.backward()
        #print([np.sum(_.grad.data.numpy()) for _ in net.parameters()])
        for param in net.parameters():
            param.grad[param.grad != param.grad] = 0
        #print("modified.")
        #print([np.sum(_.grad.data.numpy()) for _ in net.parameters()])
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm(net.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad()

        if save_every is not None:
            if time.time() - last_save > save_every:
                print("Saving..")
                torch.save(net.state_dict(), weights_filename)
                last_save = time.time()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        usage()
        sys.exit()
    elif sys.argv[1] in ("-h", "--help", "?"):
        usage()
        sys.exit()
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
