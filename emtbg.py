#Only using the 'go' action


import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import torch.multiprocessing as mp
import os.path
import os
import sys
import time
from operator import itemgetter
import six
assert six.PY3
from env import Game
from utils import props, notify
import jingweiz

###### CONFIG #########
target_q_ts = 60 * 5

test_epsilon = 0
test_text = False

learning_rate = 5e-4
#epsilon = 0.2
epsilon1 = 0.5
epsilon2 = 0.1
save_every = 60 * 10
notify_every = 60 * 10
#gamma = 0.5
gamma = 0.9
name="pb.pkl"
n_cpu = int(mp.cpu_count()) / 2
n_cpu = 60
#######################

embedding_dim = 20
hidden_size = 20
hidden_size2 = 20


Net = None


class Net_dnc(nn.Module):
    def __init__(self, num_symbols, num_actions, num_objects):
        super(Net_dnc, self).__init__()
        self.num_symbols = num_symbols
        self.embedding = nn.Embedding(num_symbols, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)

        class Empty():
            use_cuda = False
            dtype = torch.FloatTensor
        args = Empty()
        args.batch_size = 1
        args.input_dim = hidden_size
        args.output_dim = hidden_size
        args.hidden_dim = 64
        args.num_write_heads = 1
        args.num_read_heads = 4
        args.mem_hei = 16
        args.mem_wid = 16
        args.clip_value = 20.
        args.controller_params = Empty()
        args.accessor_params = Empty()
        args.accessor_params.write_head_params = Empty()
        args.accessor_params.write_head_params.num_allowed_shifts = 3
        args.accessor_params.read_head_params = Empty()
        args.accessor_params.read_head_params.num_allowed_shifts = 3
        args.accessor_params.memory_params = Empty()
        self.circuit = jingweiz.DNCCircuit(args)

        self.linear = nn.Linear(hidden_size + hidden_size, hidden_size2)
        self.objects = nn.Linear(hidden_size2, num_objects)


    def reset(self):
        self.circuit._reset()


    def forward(self, x):
        x = x - 1
        x2 = self.embedding(x)

        mask = x != self.num_symbols - 1
        mask = mask.float()
        mask = mask.unsqueeze(2)
        x2 = x2 * mask

        x3, _ = self.lstm(x2)
        x4 = torch.sum(x3, 1)

        x4b = self.circuit(x4)
        x4b = x4b.view((1,-1))

        x4c = torch.cat((x4, x4b), 1)
        x5 = F.relu(self.linear(x4c))
        return self.objects(x5)



class Net_none(nn.Module):
    def __init__(self, num_symbols, num_actions, num_objects):
        super(Net_none, self).__init__()
        self.num_symbols = num_symbols
        self.embedding = nn.Embedding(num_symbols, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)

        self.linear = nn.Linear(hidden_size, hidden_size2)
        self.objects = nn.Linear(hidden_size2, num_objects)


    def reset(self):
        pass


    def forward(self, x):
        x = x - 1
        x2 = self.embedding(x)

        mask = x != self.num_symbols - 1
        mask = mask.float()
        mask = mask.unsqueeze(2)
        x2 = x2 * mask

        x3, _ = self.lstm(x2)
        x4 = torch.sum(x3, 1)

        x5 = F.relu(self.linear(x4))
        return self.objects(x5)



class Net_avg(nn.Module):
    def __init__(self, num_symbols, num_actions, num_objects):
        super(Net_avg, self).__init__()
        self.num_symbols = num_symbols
        self.embedding = nn.Embedding(num_symbols, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)

        self.linear = nn.Linear(hidden_size + hidden_size, hidden_size2)
        self.objects = nn.Linear(hidden_size2, num_objects)


    def reset(self):
        self.avg = np.zeros((1, hidden_size))


    def forward(self, x):
        x = x - 1
        x2 = self.embedding(x)

        mask = x != self.num_symbols - 1
        mask = mask.float()
        mask = mask.unsqueeze(2)
        x2 = x2 * mask

        x3, _ = self.lstm(x2)
        x4 = torch.sum(x3, 1)

        x4b = Variable(torch.Tensor(self.avg))

        self.avg = self.avg + x4.data.numpy()

        x4c = torch.cat((x4, x4b), 1)
        x5 = F.relu(self.linear(x4c))
        return self.objects(x5)



class Net_lstm(nn.Module):
    def __init__(self, num_symbols, num_actions, num_objects):
        super(Net_lstm, self).__init__()
        self.num_symbols = num_symbols
        self.embedding = nn.Embedding(num_symbols, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)

        self.cell = nn.LSTMCell(hidden_size, hidden_size)

        self.linear = nn.Linear(hidden_size + hidden_size, hidden_size2)
        self.objects = nn.Linear(hidden_size2, num_objects)


    def reset(self):
        self.cx = Variable(torch.Tensor(np.zeros((1, hidden_size))))
        self.hx = Variable(torch.Tensor(np.zeros((1, hidden_size))))


    def forward(self, x):
        x = x - 1
        x2 = self.embedding(x)

        mask = x != self.num_symbols - 1
        mask = mask.float()
        mask = mask.unsqueeze(2)
        x2 = x2 * mask

        x3, _ = self.lstm(x2)
        x4 = torch.sum(x3, 1)

        self.hx, self.cx = self.cell(x4, (self.hx, self.cx))

        x4b = self.hx
        #x4b = x4b.view((1,-1))

        x4c = torch.cat((x4, x4b), 1)
        x5 = F.relu(self.linear(x4c))
        return self.objects(x5)



def train(net, rank):
    torch.set_num_threads(1)  #also do: export MKL_NUM_THREADS=1

    net.reset()
    env = Game(True, 4000 + rank + 1, max_steps=250)

    target_net = Net(1254, 6, 36)
    target_net.load_state_dict(net.state_dict())
    target_net.reset()

    epsilon = epsilon1

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
    last_save = time.time()
    last_notify = time.time()
    last_sync = time.time()
    episode_number = 0
    terminal = True
    prev_value = None
    available_objects = None
    num_objects = len(env.objects)
    recent_rewards_of_episodes = []
    recent_steps_of_episodes = []

    quest1_reward_cnt = 0
    quest2_reward_cnt = 0
    quest3_reward_cnt = 0
    quest4_reward_cnt = 0
    quest1_rewards = np.zeros(100)
    quest2_rewards = np.zeros(100)
    quest3_rewards = np.zeros(100)
    quest4_rewards = np.zeros(100)

    while True:
        if terminal:
            student_saw_obelisk = False
            quest1_rewards[episode_number % len(quest1_rewards)] = 0
            quest2_rewards[episode_number % len(quest2_rewards)] = 0
            quest3_rewards[episode_number % len(quest3_rewards)] = 0
            quest4_rewards[episode_number % len(quest4_rewards)] = 0
            prev_value = None
            num_steps = 0
            net.reset()
            target_net.reset()
            state, reward, terminal, available_objects = env.reset()
            sum_rewards = reward

        state = torch.LongTensor(state)
        objects_probs = net(Variable(state.unsqueeze(0)))

        _objects_probs = objects_probs.data.numpy()

        #Choose action
        if random.random() < epsilon:
            if available_objects is None:
                objects = list(enumerate(env.objects))
            else:
                objects = [_ for _ in list(enumerate(env.objects)) if _[0] in available_objects]

            _object = random.choice(objects)[0]
        else:
            if available_objects is not None:
                mask = np.zeros(num_objects)
                for e in available_objects:
                    mask[e] = 1
                _objects_probs = objects_probs.data.numpy() * mask
                _objects_probs = _objects_probs + (_objects_probs == 0) * -1e30
            _object = int(np.argmax(_objects_probs))

        prev_value = objects_probs[0, _object]

        # step the environment and get new measurements
        state, reward, terminal, available_objects = env.step(5, _object)
        sum_rewards += reward
        num_steps += 1

        if reward > 10 - 0.0001:
            quest4_reward_cnt = quest4_reward_cnt + 1
            quest4_rewards[episode_number % len(quest4_rewards)] = 1
        elif reward > 8 - 0.0001:
            quest3_reward_cnt = quest3_reward_cnt + 1
            quest3_rewards[episode_number % len(quest3_rewards)] = 1
            if not student_saw_obelisk:
                reward = -8
                terminal = True
        elif reward > 7 - 0.0001:
            student_saw_obelisk = True
            quest2_reward_cnt = quest2_reward_cnt + 1
            quest2_rewards[episode_number % len(quest2_rewards)] = 1
            if np.mean(quest2_rewards) < 0.75 and random.random() < 0.9:
                terminal = True
        elif reward > 5 - 0.0001:
            quest1_reward_cnt = quest1_reward_cnt + 1
            quest1_rewards[episode_number % len(quest1_rewards)] = 1
            if np.mean(quest1_rewards) < 0.9 and random.random() < 0.85:
                terminal = True

        if 2 * epsilon > (epsilon1 + epsilon2):
            if np.mean(quest3_rewards) > .98:
                if np.mean(quest2_rewards) > .98:
                    if np.mean(quest1_rewards) > .98:
                        epsilon = epsilon2
                        if rank == 0:
                            notify("Epsilon is now:", epsilon)


        if terminal:
            next_value = 0
        else:
            if target_q_ts is None:
                next_value = float(np.max(_objects_probs))
            else:
                state = torch.LongTensor(state)
                objects_probs = target_net(Variable(state.unsqueeze(0)))
                _objects_probs = objects_probs.data.numpy()
                if available_objects is not None:
                    mask = np.zeros(num_objects)
                    for e in available_objects:
                        mask[e] = 1
                    _objects_probs = _objects_probs * mask
                    _objects_probs = _objects_probs + (_objects_probs == 0) * -1e30
                next_value = float(np.max(_objects_probs))

        loss = (reward + gamma * next_value - prev_value) ** 2

        #Update for only a tenth of the non important steps
        if abs(reward) > 4 or random.random() < 0.05:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm(net.parameters(), 1)
            optimizer.step()

        if terminal:
            recent_rewards_of_episodes.append(sum_rewards)
            recent_steps_of_episodes.append(num_steps)
            if len(recent_rewards_of_episodes) > 100:
                recent_rewards_of_episodes.pop(0)
            if len(recent_steps_of_episodes) > 100:
                recent_steps_of_episodes.pop(0)

            episode_number += 1
            if target_q_ts is not None and time.time() - last_sync > target_q_ts:
                if rank == 0:
                    print("Update target")
                target_net.load_state_dict(net.state_dict())
                last_sync = time.time()

            if rank == 0:
                summary = "{} {:.4} {} {:.4} {:.4} Qc: {} {} {} {} Q: {} {} {} {}".format(episode_number, sum_rewards, num_steps, np.mean(recent_rewards_of_episodes), np.mean(recent_steps_of_episodes), quest1_reward_cnt, quest2_reward_cnt, quest3_reward_cnt, quest4_reward_cnt, np.mean(quest1_rewards), np.mean(quest2_rewards), np.mean(quest3_rewards), np.mean(quest4_rewards))
                print(summary)


                if save_every is not None:
                    if time.time() - last_save > save_every:
                        print("Saving..")
                        torch.save(net.state_dict(), name)
                        last_save = time.time()

                if notify_every is not None:
                    if time.time() - last_notify > notify_every:
                        print("Notify..")
                        notify(summary)
                        last_notify = time.time()


def test(net, env, is_tutorial_world):
    num_objects = len(env.objects)

    quest4_reward_cnt = 0
    quest3_reward_cnt = 0
    quest2_reward_cnt = 0
    quest1_reward_cnt = 0
    total_reward = 0
    nrewards = 0
    nepisodes = 0
    episode_reward = 0
    total_steps = 0
    num_step = 0

    terminal = True
    available_objects = None
    while True:
        if terminal:
            num_step = 0
            if test_text:
                print("Press enter to start new game:")
                input()
            net.reset()
            state, reward, terminal, available_objects = env.reset()
            if test_text:
                print(env.state2text(state, reward))

        state = torch.LongTensor(state)
        objects_probs = net(Variable(state.unsqueeze(0)))

        if test_text:
            print("Actions:", list(enumerate(env.actions)))
            if available_objects is None:
                print("Objects:", list(enumerate(env.objects)))
            else:
                print("Objects:", [_ for _ in list(enumerate(env.objects)) if _[0] in available_objects])

        #Choose action
        if random.random() < test_epsilon:
            actions = list(enumerate(env.actions))
            if available_objects is None:
                objects = list(enumerate(env.objects))
            else:
                objects = [_ for _ in list(enumerate(env.objects)) if _[0] in available_objects]

            _action = 5
            _object = random.choice(objects)[0]
            if test_text:
                print(">>> " + str(_action) + "." + env.actions[_action] + " " + str(_object) + "." + env.objects[_object] + " [random choice]")
        else:
            _action = 5
            if available_objects is not None:
                mask = np.zeros(num_objects)
                for e in available_objects:
                    mask[e] = 1
                mask = torch.Tensor(mask)
                mask = mask.unsqueeze(0)
                mask = Variable(mask)
                objects_probs = objects_probs * mask
                objects_probs = objects_probs + (objects_probs == 0).float() * -1e30
            _object = int(np.argmax(objects_probs.data.numpy()))
            if test_text:
                print("["+str(num_step)+"]>>> " + str(_action) + "." + env.actions[_action] + " " + str(_object) + "." + env.objects[_object])
                print()

        state, reward, terminal, available_objects = env.step(_action, _object)
        if test_text:
            print(env.state2text(state, reward))
        total_steps += 1
        num_step += 1

        if is_tutorial_world:
            if reward > 10 - 0.0001:
                quest4_reward_cnt = quest4_reward_cnt + 1
            elif reward > 8 - 0.0001:
                quest3_reward_cnt = quest3_reward_cnt + 1
            elif reward > 7 - 0.0001:
                quest2_reward_cnt = quest2_reward_cnt + 1
            elif reward > 5 - 0.0001:
                quest1_reward_cnt = quest1_reward_cnt + 1
        else:
            if reward > 0.9:
                quest1_reward_cnt = quest1_reward_cnt + 1

        episode_reward = episode_reward + reward
        if reward != 0:
           nrewards = nrewards + 1

        if terminal:
            total_reward = total_reward + episode_reward
            episode_reward = 0
            nepisodes = nepisodes + 1

            if is_tutorial_world:
                print(nepisodes, "avg steps:", total_steps / nepisodes, "avg reward:", total_reward / nepisodes, "Q1:", quest1_reward_cnt / nepisodes, "Q2:", quest2_reward_cnt / nepisodes, "Q3:", quest3_reward_cnt / nepisodes, "Q4:", quest4_reward_cnt / nepisodes)
            else:
                print(nepisodes, "avg steps:", total_steps / nepisodes, "avg reward:", total_reward / nepisodes, "Q1:", quest1_reward_cnt / nepisodes)


def main():
    global Net
    #global epsilon
    global n_cpu

    port = 4001
    test_mode = False
    Net = Net_dnc

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        for c in list(cmd):
            if c == 't':
                test_mode = True
            elif c == 'T':
                test_mode = True
                test_text = True
            elif c == 'p':
                port = int(sys.argv[2])
                print("port:", port)
            #elif c == 'e':
            #    epsilon = float(sys.argv[2])
            #    print("epsilon:", epsilon)
            elif c == 'd':
                Net = Net_dnc
            elif c == 'l':
                Net = Net_lstm
            elif c == 'a':
                Net = Net_avg
            elif c == 'n':
                Net = Net_none
            elif c == '1':
                n_cpu = 1

    print("Using:", Net)

    #net = Net(len(env.symbols), len(env.actions), len(env.objects))
    net = Net(1254, 6, 36)

    #Try to load from file
    if os.path.isfile(name):
        print("Loading from file..")
        net.load_state_dict(torch.load(name))


    if not test_mode:
        net.share_memory()
        processes = []
        for rank in range(n_cpu):
            p = mp.Process(target=train, args=(net, rank))
            p.start()
            processes.append(p)
        for p in processes:
          p.join()
    else:
        env = Game(True, port, max_steps=250)
        test(net, env, True)


if __name__ == "__main__":
    main()
