""" Some API to make tensorwolf look like tensorflow """
from tensorwolf.executor import *
zeros = np.zeros
ones = np.ones
float32 = np.float32


class Session(object):
    def __call__(self, name="Session"):
        """ Just a shell, nothing else."""
        newSession = Session()
        newSession.name = name
        newSession.ex = None
        return newSession

    def run(self, eval_node_list, feed_dict={}):
        isList = True
        if not isinstance(eval_node_list, list):
            isList = False
            eval_node_list = [eval_node_list]
        self.ex = Executor(eval_node_list=eval_node_list)
        if isList:
            return self.ex.run(feed_dict=feed_dict)
        else:
            return self.ex.run(feed_dict=feed_dict)[0]

    def __enter__(self):
        return self

    def __exit__(self, e_t, e_v, t_b):
        # I do not know what these args mean...
        return
