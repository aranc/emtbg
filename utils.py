import pexpect
import numpy as np
import os
import os.path

def notify(msg):
    if os.path.exists(os.path.expanduser("~/arantgbot")): #Aran: This send notifications to my phone, it will not work for other people
        if not b"laptop" in pexpect.run('hostname'):
            pexpect.run('bash -c "~/arantgbot/tg_sendmsg.py [`hostname`] ' + msg + '"')


def props(x, e):
    p = x.count(e)
    n = len(x)
    return "%2.2f" % (float(p) / float(n))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)
