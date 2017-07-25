""" Some API to make tensorwolf look like tensorflow """
from tensorwolf.executor import *


class Session(object):
    def __call__(self, name="Session"):
        """ Just a shell, nothing else."""
        newSession = Session()
        newSession.name = name
        newSession.ex = None
        return newSession

    def run(self, eval_node_list, feed_dict={}):
        if not isinstance(eval_node_list, list):
            eval_node_list = [eval_node_list]
        self.ex = Executor(eval_node_list=eval_node_list)
        return self.ex.run(feed_dict=feed_dict)[0]

    def __enter__(self):
        return self

    def __exit__(self, e_t, e_v, t_b):
        # I do not know what these args mean...
        return
