import numpy as np
import random


###### CONFIG #########
vector_len = 6
symbol_len = 3
seq_min_len = 2
seq_max_len = 6
#######################

def generate_new_symbol():
    return tuple(random.randint(0, 1) for _ in range(vector_len * symbol_len))

def symbol_to_matrix(symbol):
    ret = np.zeros((vector_len + 2, symbol_len + 1))
    ret[:-2,1:] = np.array(symbol).reshape(vector_len, symbol_len)
    ret[-2,0] = 1
    return ret

class Challenge:
    def get_in_vector_len(self):
        #plus one for a control channel
        return vector_len + 2

    def get_out_vector_len(self):
        return vector_len

    def get_next_task(self):
        seq_len = random.randint(seq_min_len, seq_max_len)
        keys = []
        while len(keys) < seq_len:
            new_symbol = generate_new_symbol()
            if new_symbol not in keys:
                keys.append(new_symbol)
        _dict = {key:generate_new_symbol() for key in keys}
        
        pairs = [(symbol_to_matrix(key), symbol_to_matrix(_dict[key])) for key in keys]
        flatten = [item for sublist in pairs for item in sublist]
        cat = np.hstack(flatten)
        assert cat.shape == (vector_len + 2, (symbol_len + 1) * seq_len * 2)

        question = random.choice(keys)

        _question = symbol_to_matrix(question)
        _question[-2,0] = 0
        _question[-1,0] = 1
        _input = np.hstack((cat, _question, _question[:,0].reshape((-1,1))))
        assert _input.shape == (vector_len + 2, (symbol_len + 1) * (seq_len * 2 + 1) + 1)

        return _input, symbol_to_matrix(_dict[question])[:-2,1:]


def get_challenge():
    return Challenge()
