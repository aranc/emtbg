import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import os.path
import os
import sys
import time
from operator import itemgetter
import six
assert six.PY3
from lp import Game
from utils import props, notify

###### CONFIG #########
test_epsilon = 0.05
target_q = 1000
test_text = False

exp_max_len = 100000
batch_size = 64
p_priority = 0.5
exp_min_len = 1000

learning_rate = 5e-4
epsilon = 0.2
save_every = 60
notify_every = 60 * 10
gamma = 0.5
name="pa.pkl"
#######################
#lr=0.0005,ep=1,ep_end=0.2,ep_endt=200000,discount=0.5,hist_len=1,learn_start=1000,replay_memory=100000,update_freq=4,n_replay=1,minibatch_size=64,rescale_r=1,ncols=1,bufferSize=512,valid_size=500,target_q=1000,clip_delta=1,min_reward=-1,max_reward=10

#batch_size = 4
#exp_min_len = 20
#exp_max_len = 50


embedding_dim = 20
hidden_size = 20
hidden_size2 = 20
class Net(nn.Module):
    def __init__(self, num_symbols, num_actions, num_objects):
        super(Net, self).__init__()
        self.num_symbols = num_symbols
        self.embedding = nn.Embedding(num_symbols, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size2)
        self.actions = nn.Linear(hidden_size2, num_actions)
        self.objects = nn.Linear(hidden_size2, num_objects)


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
        return self.actions(x5), self.objects(x5)



def train(net, env):
    if target_q is None:
        target_net = None
    else:
        target_net = Net(len(env.symbols), len(env.actions), len(env.objects))
        target_net.load_state_dict(net.state_dict())

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
    last_save = time.time()
    last_notify = time.time()
    episode_number = 0
    terminal = True
    prev_value = None
    available_objects = None
    num_objects = len(env.objects)
    recent_rewards_of_episodes = []
    recent_steps_of_episodes = []

    exp_prev_states = np.zeros((exp_max_len, env.state_dim), dtype=int)
    exp_actions = np.zeros(exp_max_len)
    exp_objects = np.zeros(exp_max_len)
    exp_next_values = np.zeros(exp_max_len)
    exp_rewards = np.zeros(exp_max_len)
    exp_priority = np.zeros(exp_max_len, dtype=np.uint8)
    exp_terminals = np.zeros(exp_max_len, dtype=np.uint8)
    exp_idx = 0
    exp_filled = False

    while True:
        if terminal:
            prev_value = None
            num_steps = 0
            state, reward, terminal, available_objects = env.reset()
            sum_rewards = reward

        state = torch.LongTensor(state)
        actions_probs, objects_probs = net(Variable(state.unsqueeze(0)))

        #Choose action
        if ((not exp_filled) and exp_idx < exp_min_len) or random.random() < epsilon:
            actions = list(enumerate(env.actions))
            if available_objects is None:
                objects = list(enumerate(env.objects))
            else:
                objects = [_ for _ in list(enumerate(env.objects)) if _[0] in available_objects]

            _action = random.choice(actions)[0]
            _object = random.choice(objects)[0]
        else:
            _action = int(np.argmax(actions_probs.data.numpy()))
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

        prev_state = state

        # step the environment and get new measurements
        state, reward, terminal, available_objects = env.step(_action, _object)
        sum_rewards += reward
        num_steps += 1

        if terminal:
            next_value = 0
        else:
            if target_q is None:
                next_value = float(np.max(actions_probs.data.numpy()) + np.max(objects_probs.data.numpy()))
            else:
                actions_probs, objects_probs = target_net(Variable(torch.LongTensor(state).unsqueeze(0)))
                if available_objects is not None:
                    mask = np.zeros(num_objects)
                    for e in available_objects:
                        mask[e] = 1
                    mask = torch.Tensor(mask)
                    mask = mask.unsqueeze(0)
                    mask = Variable(mask)
                    objects_probs = objects_probs * mask
                    objects_probs = objects_probs + (objects_probs == 0).float() * -1e30
                next_value = float(np.max(actions_probs.data.numpy()) + np.max(objects_probs.data.numpy()))

        exp_prev_states[exp_idx] = prev_state
        exp_actions[exp_idx] = _action
        exp_objects[exp_idx] = _object
        exp_next_values[exp_idx] = next_value
        exp_rewards[exp_idx] = reward
        exp_priority[exp_idx] = 1 if reward > 0 else 0
        exp_terminals[exp_idx] = 1 if terminal else 0
        if exp_idx == exp_max_len - 1:
            exp_filled = True
        exp_idx = (exp_idx + 1) % exp_max_len

        if target_net is not None and exp_idx % target_q == 1:
            print("copy target net")
            target_net.load_state_dict(net.state_dict())

        if (exp_idx % 4 == 3) and (exp_filled or exp_idx > exp_min_len):
            exp_current_size = exp_max_len if exp_filled else exp_idx
            pos_exp = [_ for _ in  range(exp_current_size) if exp_priority[_] == 1]
            batch = np.zeros(batch_size, dtype=np.int)
            for i in range(batch_size):
                p = random.random()
                if p <= p_priority and len(pos_exp) > 0:
                    t = random.choice(pos_exp)
                else:
                    t = random.randint(0, exp_current_size - 1)
                batch[i] = t

            prev_states = np.take(exp_prev_states, batch, axis=0)
            _actions = np.take(exp_actions, batch, axis=0)
            _objects = np.take(exp_objects, batch, axis=0)
            rewards = np.take(exp_rewards, batch, axis=0)
            terminals = np.take(exp_terminals, batch, axis=0)
            next_values = np.take(exp_next_values, batch, axis=0)

            actions_probs, objects_probs = net(Variable(torch.LongTensor(prev_states)))
            
            _actions_probs = torch.gather(actions_probs, 1, Variable(torch.LongTensor(_actions).unsqueeze(1)))
            _objects_probs = torch.gather(objects_probs, 1, Variable(torch.LongTensor(_objects).unsqueeze(1)))
            prev_values = _actions_probs + _objects_probs
            prev_values = prev_values.squeeze()
            
            rewards = Variable(torch.Tensor(rewards))
            terminals = Variable(torch.Tensor(terminals))
            next_values = Variable(torch.Tensor(next_values))

            loss = (rewards + (gamma * next_values) * (1 - terminals) - prev_values) ** 2
            loss = torch.sum(loss)

            optimizer.zero_grad()
            loss.backward()
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

            summary = "[{}{}] {} {} {} {} {}".format("F" if exp_filled else "E", exp_idx, episode_number, sum_rewards, num_steps, np.mean(recent_rewards_of_episodes), np.mean(recent_steps_of_episodes))
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

    quest3_reward_cnt = 0
    quest2_reward_cnt = 0
    quest1_reward_cnt = 0
    total_reward = 0
    nrewards = 0
    nepisodes = 0
    episode_reward = 0
    total_steps = 0

    terminal = True
    available_objects = None
    while True:
        if terminal:
            if test_text:
                print("Press enter to start new game:")
                input()
            state, reward, terminal, available_objects = env.reset()
            if test_text:
                print(env.state2text(state, reward))

        state = torch.LongTensor(state)
        actions_probs, objects_probs = net(Variable(state.unsqueeze(0)))

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

            _action = random.choice(actions)[0]
            _object = random.choice(objects)[0]
            if test_text:
                print(">>> " + str(_action) + "." + env.actions[_action] + " " + str(_object) + "." + env.objects[_object] + " [random choice]")
        else:
            _action = int(np.argmax(actions_probs.data.numpy()))
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
                print(">>> " + str(_action) + "." + env.actions[_action] + " " + str(_object) + "." + env.objects[_object])

        state, reward, terminal, available_objects = env.step(_action, _object)
        if test_text:
            print(env.state2text(state, reward))
        total_steps += 1

        if is_tutorial_world:
            if reward > 4:
                quest1_reward_cnt =quest1_reward_cnt+1
            elif reward > 9:
                quest2_reward_cnt = quest2_reward_cnt + 1
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
                print(nepisodes, "avg steps:", total_steps / nepisodes, "avg reward:", total_reward / nepisodes, "Q1:", quest1_reward_cnt / nepisodes, "Q2:", quest2_reward_cnt / nepisodes)
            else:
                print(nepisodes, "avg steps:", total_steps / nepisodes, "avg reward:", total_reward / nepisodes, "Q1:", quest1_reward_cnt / nepisodes)


def main():
    is_test = False
    is_tutorial_world = False
    port = 4001
    if len(sys.argv) > 1:
        assert(sys.argv[1] in ('master', 'fantasy', 'mastert', 'fantasyt'))
        is_tutorial_world = sys.argv[1] in ('fantasy', 'fantasyt')
        is_test = sys.argv[1] in ('mastert', 'fantasyt')
    if len(sys.argv) > 2:
        port = int(sys.argv[2])

    env = Game(is_tutorial_world, port)

    net = Net(len(env.symbols), len(env.actions), len(env.objects))
    #Try to load from file
    if os.path.isfile(name):
        print("Loading from file..")
        net.load_state_dict(torch.load(name))

    if is_test:
        test(net, env, is_tutorial_world)
    else:
        train(net, env)


if __name__ == "__main__":
    main()
