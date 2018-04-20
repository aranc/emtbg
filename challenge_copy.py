import numpy as np
import random


###### CONFIG #########
vector_len = 8
seq_min_len = 2
seq_max_len = 2
#######################


class Challenge:
    def get_in_vector_len(self):
        #plus one for a control channel
        return vector_len + 1

    def get_out_vector_len(self):
        return vector_len

    def get_next_task(self):
        seq_len = random.randint(seq_min_len, seq_max_len)
        mat = np.random.randint(0, 2, (vector_len + 1, seq_len+1)).astype(np.float64)
        mat[-1,:] = 0
        mat[:,-1] = 0
        mat[-1, -1] = 1
        return mat, mat[:-1,:-1]
        return mat, np.array([0,1,0,0,0,0,1,0]).reshape(8,1)


def get_challenge():
    return Challenge()
