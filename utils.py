import pexpect
import numpy as np

def notify(msg):
    if not b"laptop" in pexpect.run('hostname'):
        pexpect.run('bash -c "~/arantgbot/tg_sendmsg.py [`hostname`] ' + msg + '"')


def props(x, e):
    p = x.count(e)
    n = len(x)
    return "%2.2f" % (float(p) / float(n))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)
