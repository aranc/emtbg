import readline
import sys
import lutorpy as lua

def table2list(table):
    return [table[i] for i in range(len(table))]


class Game:
    def __init__(self, is_tutorial_world, port=4001, max_steps=None):
        self.is_tutorial_world = is_tutorial_world

        self.lp = lua.LuaRuntime(unpack_returned_tuples=True, zero_based_index=True)
        self.lp.execute("QUEST_LEVELS=1")
        if is_tutorial_world:
            self.max_steps = 250
        else:
            self.max_steps = 20
        if max_steps is not None:
            self.max_steps = max_steps
        self.lp.execute("MAX_STEPS="+str(self.max_steps))
        self.lp.execute("RECURRENT=1")
        if is_tutorial_world:
            self.state_dim = 100
        else:
            self.state_dim = 30
        self.lp.execute("STATE_DIM="+str(self.state_dim))
        self.lp.require("underscore")
        self.lp.require("io")
        self.lp.require("torch")
        self.lp.require("xlua")
        self.lp.require("utils")
        self.lp.require("client")

        if not self.is_tutorial_world:
            self.framework = self.lp.require("framework")
        else:
            self.framework = self.lp.require("framework_fantasy")

        self.lp.eval("client_connect('"+str(port)+"')")
        self.lp.eval("login('root', 'root')")

        if not self.is_tutorial_world:
            self.framework["makeSymbolMapping"]('../text-world/evennia/contrib/text_sims/build.ev')
        else:
            self.framework["makeSymbolMapping"]('../text-world/evennia/contrib/tutorial_world/build.ev')

        self.symbols = table2list(self.lp.eval("symbols"))
        self.symbols.append(None)
        self.actions = table2list(self.framework["getActions"]())
        self.objects = table2list(self.framework["getObjects"]())


    def _parse_table(self, table):
        table = list(table)
        if len(table) == 3:
            table.append(None)
        state, reward, terminal, available_objects = table
        state = state.asNumpyArray()
        if available_objects is not None:
            available_objects = table2list(available_objects)
            available_objects = [_ - 1 for _ in available_objects]
        return state, reward, terminal, available_objects


    def reset(self):
        table = self.framework["newGame"]()
        return self._parse_table(table)

    
    def step(self, _action, _object):
        table = self.framework["step"](_action + 1, _object + 1, None)
        return self._parse_table(table)


    def state2text(self, state, reward = None):
        text = [self.symbols[int(_ - 1)] for _ in state]
        text = " ".join([_ for _ in reversed(text) if _ is not None])
        if reward is not None:
            text = "[" + str(reward) + "] " + text
        return text

if __name__ == "__main__":
    is_tutorial_world = False
    port = 4001
    if len(sys.argv) > 1:
        assert(sys.argv[1] in ('master', 'fantasy'))
        is_tutorial_world = sys.argv[1] == 'fantasy'
    if len(sys.argv) > 2:
        port = int(sys.argv[2])

    env = Game(is_tutorial_world, port)
    print("Actions:", list(enumerate(env.actions)))
    print("Objects:", list(enumerate(env.objects)))
    terminal = True
    available_objects = None
    while True:
        if terminal:
            print()
            print()
            print("##### NEW GAME #####")
            state, reward, terminal, available_objects = env.reset()
            print(env.state2text(state, reward))

        print("Actions:", list(enumerate(env.actions)))
        if available_objects is None:
            print("Objects:", list(enumerate(env.objects)))
        else:
            print("Objects:", [_ for _ in list(enumerate(env.objects)) if _[0] in available_objects])
        command = input()
        if command == "new":
            terminal = True
            continue
        command = command.split()
        if len(command) != 2:
            print("Bad command len")
            continue
        if [str(int(_)) for _ in command] != command:
            print("Bad command values")
            continue
        command = [int(_) for _ in command]
        if command[0] < 0 or command[1] < 0 or command[0] >= len(env.actions) or command[1] >= len(env.objects):
            print("Bad command range")
            continue

        state, reward, terminal, available_objects = env.step(command[0], command[1])
        print()
        print(env.state2text(state, reward))


