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
from utils import props, notify
import jingweiz

###### CONFIG #########
max_episodes = None
max_episodes = 1000

target_q_ts = 60 * 5

test_epsilon = 0
test_text = False

learning_rate = 5e-4
epsilon = 0.1
epsilon_annealing = False
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_length = 2000

save_every = 60 * 10
notify_every = 60 * 10
save_every_episodes = 100
save_every_episodes = None
#gamma = 0.5
gamma = 0.9
gamma_tmaze = 0.99
name="weights.pkl"
name_stats="stats.pkl"
n_cpu = int(mp.cpu_count())
#######################


max_desc_size = 30
max_steps = 15
max_steps_tmaze = 25


Game = None


class Tunnel():
    length = 5

    win_reward = 1.0
    loss_reward = -1.0

    _words = None
    _symbols_table = None

    def __init__(self):
        self.words()
        self.text2indices("")


    def reset(self):
        self.start = random.choice((0, self.length))
        self.end = self.length - self.start
        self.pos = self.start
        return self._pos2text(), 0, False, self._meta_info()

    def _meta_info(self):
        return "{} ({})".format(self.pos, self.end)

    def _pos2text(self):
        if self.pos == self.end:
            return "You have reached the end of the tunnel"
        if self.pos == 0:
            return "You are in a tunnel, there is wall to your left"
        if self.pos == self.length:
            return "You are in a tunnel, there is wall to your right"
        return "You are in a tunnel, you may go both ways"

    def actions(self):
        return "right", "left"

    def words(self):
        if self._words is None:
            _words = []
            _words += "You have reached the end of the tunnel".replace(",", "").split()
            _words += "You are in a tunnel, there is wall to your left".replace(",", "").split()
            _words += "You are in a tunnel, there is wall to your right".replace(",", "").split()
            _words += "You are in a tunnel, you may go both ways".replace(",", "").split()
            self._words = set(_words)
        return self._words

    def text2indices(self, text):
        if self._symbols_table is None:
            words = sorted(self.words())
            words = enumerate(words)
            table = {}
            for idx, sym in words:
                table[sym] = idx
            self._symbols_table = table
        
        text = text.replace(",", "").split()
        text = [self._symbols_table[word] for word in text]
        if len(text) < max_desc_size:
            empty = len(self._words)
            prefix = [empty] * (max_desc_size - len(text))
            text = prefix + text
        assert len(text) == max_desc_size
        return text

    def step(self, action):
        reward = -0.01
        terminal = False

        if action == 0:
            self.pos = self.pos + 1
        if action == 1:
            self.pos = self.pos - 1
        if self.pos < 0:
            #reward = -0.1
            self.pos = 0
        if self.pos > self.length:
            #reward = -0.1
            self.pos = self.length

        if self.pos == self.end:
            terminal = True
            reward = self.win_reward

        return self._pos2text(), reward, terminal, self._meta_info()

        def apply_curriculum(self, episode_number):
            pass


class Tmaze():
    length = 1

    win_reward = 1.0
    loss_reward = -1.0
    default_reward = 0.0

    _words = None
    _symbols_table = None

    def __init__(self):
        self.words()
        self.text2indices("")


    def reset(self):
        self.desired_color = random.choice(("red", "green"))
        self.upper_color = random.choice(("red", "green"))
        self.lower_color = "red" if self.upper_color == "green" else "green"
        assert self.lower_color != self.upper_color
        assert self.lower_color in ("red", "green")
        assert self.upper_color in ("red", "green")
        self.pos = 0
        return self._pos2text(), 0, False, self._meta_info()

    def apply_curriculum(self, episode_number):
        if episode_number < 2000:
            if random.random() < .9:
                self.lower_color = self.desired_color
                self.upper_color = self.desired_color
        elif episode_number < 3000:
            if random.random() < .8:
                self.lower_color = self.desired_color
                self.upper_color = self.desired_color
        elif episode_number < 4000:
            if random.random() < .7:
                self.lower_color = self.desired_color
                self.upper_color = self.desired_color
        elif episode_number < 5000:
            if random.random() < .6:
                self.lower_color = self.desired_color
                self.upper_color = self.desired_color
        elif episode_number < 6000:
            if random.random() < .5:
                self.lower_color = self.desired_color
                self.upper_color = self.desired_color
        elif episode_number < 7000:
            if random.random() < .4:
                self.lower_color = self.desired_color
                self.upper_color = self.desired_color
        elif episode_number < 8000:
            if random.random() < .3:
                self.lower_color = self.desired_color
                self.upper_color = self.desired_color
        elif episode_number < 9000:
            if random.random() < .2:
                self.lower_color = self.desired_color
                self.upper_color = self.desired_color
        elif episode_number < 10000:
            if random.random() < .1:
                self.lower_color = self.desired_color
                self.upper_color = self.desired_color


    def _meta_info(self):
        if self.desired_color == self.upper_color:
            if self.desired_color == self.lower_color:
                compass = "both"
            else:
                compass = "up"
        else:
            compass = "down"
        return "{} {} {}".format(self.pos, self.desired_color, compass)

    def _pos2text(self):
        if self.pos == 0:
            return "You are at the middle of a maze, you see a " + self.desired_color + " statue"
        if self.pos == self.length:
            return "You have reached the north wing of the maze, you see a " + self.upper_color + " door"
        if self.pos == -self.length:
            return "You have reached the south wing of the maze, you see a " + self.lower_color + " door"
        return "You are in a maze, you may go both ways"

    def actions(self):
        return "up", "down", "exit"

    def words(self):
        if self._words is None:
            _words = ["red", "green", "statue", "door"]
            _words += "You are at the middle of a maze, you see a ".replace(",", "").split()
            _words += "You have reached the north wing of the maze, you see a ".replace(",", "").split()
            _words += "You have reached the south wing of the maze, you see a ".replace(",", "").split()
            _words += "You are in a maze, you may go both ways".replace(",", "").split()
            _words += "You won".replace(",", "").split()
            _words += "You failed".replace(",", "").split()
            self._words = set(_words)
        return self._words

    def text2indices(self, text):
        if self._symbols_table is None:
            words = sorted(self.words())
            words = enumerate(words)
            table = {}
            for idx, sym in words:
                table[sym] = idx
            self._symbols_table = table

        text = text.replace(",", "").split()
        text = [self._symbols_table[word] for word in text]
        if len(text) < max_desc_size:
            empty = len(self._words)
            prefix = [empty] * (max_desc_size - len(text))
            text = prefix + text
        assert len(text) == max_desc_size
        return text

    def step(self, action):
        if action == 0:
            self.pos = self.pos + 1
        if action == 1:
            self.pos = self.pos - 1
        if self.pos < -self.length:
            self.pos = -self.length
        if self.pos > self.length:
            self.pos = self.length

        if action == 2 and abs(self.pos) == self.length:
            terminal = True
            current_color = self.upper_color if self.pos > 0 else self.lower_color
            won = current_color == self.desired_color
            reward = self.win_reward if won else self.loss_reward
            text = "You won" if won else "You failed"
            return text, reward, terminal, self._meta_info()

        terminal = False
        reward = self.default_reward
        return self._pos2text(), reward, terminal, self._meta_info()



embedding_dim = 20
hidden_size = 20
hidden_size2 = 20


Net = None


class Net_dnc(nn.Module):
    def __init__(self, num_symbols, num_actions):
        super(Net_dnc, self).__init__()
        self.num_symbols = num_symbols
        self.embedding = nn.Embedding(num_symbols + 1, embedding_dim)
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
        self.actions = nn.Linear(hidden_size2, num_actions)


    def reset(self):
        self.circuit._reset()


    def forward(self, x):
        x2 = self.embedding(x)

        mask = x != self.num_symbols
        mask = mask.float()
        mask = mask.unsqueeze(2)
        x2 = x2 * mask

        x3, _ = self.lstm(x2)
        x4 = torch.sum(x3, 1)

        x4b = self.circuit(x4)
        x4b = x4b.view((1,-1))

        x4c = torch.cat((x4, x4b), 1)
        x5 = F.relu(self.linear(x4c))
        return self.actions(x5)



class Net_none(nn.Module):
    def __init__(self, num_symbols, num_actions):
        super(Net_none, self).__init__()
        self.num_symbols = num_symbols
        self.embedding = nn.Embedding(num_symbols + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)

        self.linear = nn.Linear(hidden_size, hidden_size2)
        self.actions = nn.Linear(hidden_size2, num_actions)


    def reset(self):
        pass


    def forward(self, x):
        x2 = self.embedding(x)

        mask = x != self.num_symbols
        mask = mask.float()
        mask = mask.unsqueeze(2)
        x2 = x2 * mask

        x3, _ = self.lstm(x2)
        x4 = torch.sum(x3, 1)

        x5 = F.relu(self.linear(x4))
        return self.actions(x5)



class Net_avg(nn.Module):
    def __init__(self, num_symbols, num_actions):
        super(Net_avg, self).__init__()
        self.num_symbols = num_symbols
        self.embedding = nn.Embedding(num_symbols + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)

        self.linear = nn.Linear(hidden_size + hidden_size, hidden_size2)
        self.actions = nn.Linear(hidden_size2, num_actions)


    def reset(self):
        self.avg = np.zeros((1, hidden_size))


    def forward(self, x):
        x2 = self.embedding(x)

        mask = x != self.num_symbols
        mask = mask.float()
        mask = mask.unsqueeze(2)
        x2 = x2 * mask

        x3, _ = self.lstm(x2)
        x4 = torch.sum(x3, 1)

        x4b = Variable(torch.Tensor(self.avg))

        self.avg = self.avg + x4.data.numpy()

        x4c = torch.cat((x4, x4b), 1)
        x5 = F.relu(self.linear(x4c))
        return self.actions(x5)



class Net_lstm(nn.Module):
    def __init__(self, num_symbols, num_actions):
        super(Net_lstm, self).__init__()
        self.num_symbols = num_symbols
        self.embedding = nn.Embedding(num_symbols + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)

        self.cell = nn.LSTMCell(hidden_size, hidden_size)

        self.linear = nn.Linear(hidden_size + hidden_size, hidden_size2)
        self.actions = nn.Linear(hidden_size2, num_actions)


    def reset(self):
        self.cx = Variable(torch.Tensor(np.zeros((1, hidden_size))))
        self.hx = Variable(torch.Tensor(np.zeros((1, hidden_size))))


    def forward(self, x):
        x2 = self.embedding(x)

        mask = x != self.num_symbols
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
        return self.actions(x5)



def train(net, rank):
    torch.set_num_threads(1)  #also do: export MKL_NUM_THREADS=1

    env = Game()

    net.reset()
    net = Net(len(env.words()), len(env.actions()))

    target_net = Net(len(env.words()), len(env.actions()))
    target_net.load_state_dict(net.state_dict())
    target_net.reset()

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
    last_save = time.time()
    last_notify = time.time()
    last_sync = time.time()
    episode_number = 0
    terminal = True
    prev_value = None
    num_actions = len(env.actions())
    recent_rewards_of_episodes = []
    recent_steps_of_episodes = []

    win_cnt = 0
    loss_cnt = 0
    idle_cnt = 0
    recent_wins = np.zeros(100)
    recent_losses = np.zeros(100)
    recent_idles = np.zeros(100)

    if rank == 0:
        stats = []

    while True:
        if terminal:
            recent_wins[episode_number % len(recent_wins)] = 0
            recent_losses[episode_number % len(recent_losses)] = 0
            recent_idles[episode_number % len(recent_idles)] = 0
            prev_value = None
            num_steps = 0
            net.reset()
            target_net.reset()
            state, reward, terminal, meta_info = env.reset()
            env.apply_curriculum(episode_number)
            state = env.text2indices(state)
            sum_rewards = reward
            if epsilon_annealing:
                epsilon = max(epsilon_start - (epsilon_start - epsilon_end) * (episode_number / epsilon_length), epsilon_end)

        state = torch.LongTensor(state)
        actions_probs = net(Variable(state.unsqueeze(0)))

        _actions_probs = actions_probs.data.numpy()

        #Choose action
        if random.random() < epsilon:
            _action = random.choice(range(num_actions))
        else:
            _action = int(np.argmax(_actions_probs))

        prev_value = actions_probs[0, _action]

        # step the environment and get new measurements
        state, reward, terminal, meta_info = env.step(_action)
        state = env.text2indices(state)
        sum_rewards += reward
        num_steps += 1

        if num_steps == max_steps:
            terminal = True

        if terminal:
            next_value = 0
        else:
            if target_q_ts is None:
                next_value = float(np.max(_actions_probs))
            else:
                state = torch.LongTensor(state)
                actions_probs = target_net(Variable(state.unsqueeze(0)))
                _actions_probs = actions_probs.data.numpy()
                next_value = float(np.max(_actions_probs))

        loss = (reward + gamma * next_value - prev_value) ** 2

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm(net.parameters(), 1)
        optimizer.step()

        if terminal:
            if reward == env.win_reward:
                win_cnt = win_cnt + 1
                recent_wins[episode_number % len(recent_wins)] = 1
            elif reward == env.loss_reward:
                loss_cnt = loss_cnt + 1
                recent_losses[episode_number % len(recent_losses)] = 1
            else:
                idle_cnt = idle_cnt + 1
                recent_idles[episode_number % len(recent_idles)] = 1
                assert(num_steps == max_steps)

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
                stats.append({})
                stats[-1]["episode_number"] = episode_number
                stats[-1]["sum_rewards"] = sum_rewards
                stats[-1]["num_steps"] = num_steps
                stats[-1]["mean_recent_rewards_of_episodes"] = np.mean(recent_rewards_of_episodes)
                stats[-1]["mean_recent_steps_of_episodes"] = np.mean(recent_steps_of_episodes)
                stats[-1]["win_cnt"] = win_cnt
                stats[-1]["loss_cnt"] = loss_cnt
                stats[-1]["idle_cnt"] = idle_cnt
                stats[-1]["recent_wins"] = np.mean(recent_wins)
                stats[-1]["recent_losses"] = np.mean(recent_losses)
                stats[-1]["recent_idles"] = np.mean(recent_idles)
                stats[-1]["epsilon"] = epsilon

                summary = "{} {:.4} {} {:.4} {:.4} Qc: {} {} {} Q: {} {} {}".format(episode_number, sum_rewards, num_steps, np.mean(recent_rewards_of_episodes), np.mean(recent_steps_of_episodes), win_cnt, loss_cnt, idle_cnt, np.mean(recent_wins), np.mean(recent_losses), np.mean(recent_idles))
                if epsilon_annealing:
                    summary += " e: {:.4}".format(epsilon)
                print(summary)


                if save_every_episodes is not None:
                    if episode_number % save_every_episodes == 0:
                        print("Saving episodic..")
                        torch.save(net.state_dict(), str(episode_number) + "." + name)
                        with open(str(episode_number) + "." + name_stats, "wb") as _fh:
                            pickle.dump(stats, _fh)

                if save_every is not None:
                    if time.time() - last_save > save_every:
                        print("Saving..")
                        torch.save(net.state_dict(), name)
                        with open(name_stats, "wb") as _fh:
                            pickle.dump(stats, _fh)
                        last_save = time.time()

                if notify_every is not None:
                    if time.time() - last_notify > notify_every:
                        print("Notify..")
                        notify(summary)
                        last_notify = time.time()

                if max_episodes is not None and episode_number == max_episodes:
                    torch.save(net.state_dict(), name)
                    with open(name_stats, "wb") as _fh:
                        pickle.dump(stats, _fh)
                    notify(summary)
                    notify("Done.")
                    print("Done.")
                    sys.exit()


def test(net, env):
    num_actions = len(env.actions())

    win_cnt = 0
    loss_cnt = 0
    idle_cnt = 0
    total_reward = 0
    nrewards = 0
    nepisodes = 0
    episode_reward = 0
    total_steps = 0
    num_steps = 0

    terminal = True
    while True:
        if terminal:
            num_steps = 0
            if test_text:
                print("Press enter to start new game:")
                input()
                print()
                print()
                print("##### NEW GAME #####")
            net.reset()
            state, reward, terminal, meta_info = env.reset()
            if test_text:
                print("{" + str(meta_info) + "} [" + str(reward) + "] " + state)

        state = env.text2indices(state)
        state = torch.LongTensor(state)
        actions_probs = net(Variable(state.unsqueeze(0)))

        if test_text:
            print("Actions:", list(enumerate(env.actions())))

        #Choose action
        if random.random() < test_epsilon:
            _action = random.choice(range(num_actions))
            if test_text:
                print("["+str(num_steps)+"]>>> " + str(_action) + "." + env.actions[_action] + " [random choice]")
        else:
            _action = int(np.argmax(actions_probs.data.numpy()))
            if test_text:
                print("["+str(num_steps)+"]>>> " + str(_action) + "." + env.actions()[_action])
                print()

        state, reward, terminal, meta_info = env.step(_action)
        if test_text:
            print("{" + str(meta_info) + "} [" + str(reward) + "] " + state)
        total_steps += 1
        num_steps += 1

        if num_steps == max_steps:
            terminal = True

        episode_reward = episode_reward + reward
        if reward != 0:
           nrewards = nrewards + 1

        if terminal:
            total_reward = total_reward + episode_reward
            episode_reward = 0
            nepisodes = nepisodes + 1

            if reward == env.win_reward:
                win_cnt = win_cnt + 1
            elif reward == env.loss_reward:
                loss_cnt = loss_cnt + 1
            else:
                idle_cnt = idle_cnt + 1
                assert(num_steps == max_steps)

            print("{} {} {:.4} {} {:.4} {} {:.4} {} {:.4} {} {:.4}".format(nepisodes, "avg steps:", total_steps / nepisodes, "avg reward:", total_reward / nepisodes, "win:", win_cnt / nepisodes, "loss:", loss_cnt / nepisodes, "idle:", idle_cnt / nepisodes))
            if nepisodes == max_episodes:
                sys.exit()


def manual(env):
    num_actions = len(env.actions())

    terminal = True
    while True:
        if terminal:
            num_steps = 0
            print()
            print()
            print("##### NEW GAME #####")
            state, reward, terminal, meta_info = env.reset()
            print("{" + str(meta_info) + "} [" + str(reward) + "] " + state)

        print("Actions:", list(enumerate(env.actions())))

        command = input()
        if command == "new":
            terminal = True
            continue

        try:
            _action = int(command)
        except:
            print("Bad command:", command)
            continue

        state, reward, terminal, meta_info = env.step(_action)
        print("{" + str(meta_info) + "} [" + str(reward) + "] " + state)


def main():
    global Net
    global Game
    global epsilon
    global n_cpu
    global test_text
    global gamma
    global max_steps
    global epsilon_annealing

    test_mode = False
    manual_mode = False
    Net = Net_dnc
    Game = Tunnel

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        for c in list(cmd):
            if c == 't':
                test_mode = True
            elif c == 'T':
                test_mode = True
                test_text = True
            elif c == 'e':
                epsilon = float(sys.argv[2])
                print("epsilon:", epsilon)
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
            elif c == 'M':
                manual_mode = True
            elif c == 'L':
                Game = Tunnel
            elif c == 'Z':
                Game = Tmaze
                gamma = gamma_tmaze
                max_steps = max_steps_tmaze
                epsilon_annealing = True

    print("Using:", Net, Game)

    env = Game()
    net = Net(len(env.words()), len(env.actions()))

    #Try to load from file
    if os.path.isfile(name):
        print("Loading from file..")
        net.load_state_dict(torch.load(name))

    if manual_mode:
        env = Game()
        manual(env)

    elif not test_mode:
        net.share_memory()
        processes = []
        for rank in range(n_cpu):
            p = mp.Process(target=train, args=(net, rank))
            p.start()
            processes.append(p)
        for p in processes:
          p.join()
    else:
        env = Game()
        test(net, env)


if __name__ == "__main__":
    main()
