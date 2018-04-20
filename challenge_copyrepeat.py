import numpy as np
import random


###### CONFIG #########
vector_len = 8
seq_min_len = 1
seq_max_len = 10
min_repeat = 1
max_repeat = 10
#######################


class Challenge:
    def get_in_vector_len(self):
        #plus one for a control channel
        return vector_len + 1

    def get_out_vector_len(self):
        return vector_len

    def get_next_task(self):
        seq_len = random.randint(seq_min_len, seq_max_len)
        n_repeat = random.randint(min_repeat, max_repeat)
        mat = np.random.randint(0, 2, (vector_len + 1, seq_len + 1)).astype(np.float64)
        mat[-1,:] = 0
        mat[:,-1] = 0
        mat[-1, -1] = n_repeat
        return mat, np.hstack([mat[:-1,:-1]] * n_repeat)


def get_challenge():
    return Challenge()
