""" Some API to make tensorwolf look like tensorflow """
from tensorwolf.executor import *


zeros = np.zeros
ones = np.ones
float32 = np.float32
float64 = np.float64


def random_normal(shape, mean=0.0, stddev=1.0, dtype=float32, seed=None, name=None):
    return_val = np.random.normal(loc=mean, scale=stddev, size=shape)
    # print(return_val)
    return return_val


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


class train(object):
    class Optimizer(object):
        def __init__(self):
            return None

    class GradientDescentOptimizer(Optimizer):
        def __init__(self, learning_rate=1, name="GradientDescentOptimizer"):
            self.learning_rate = learning_rate
            self.name = name

        def get_variables_list(self):
            variables_list = []
            for variable in variables:
                variables_list.append(variable)
            return variables_list

        def minimize(self, target):
            variables_to_change = self.get_variables_list()
            variables_gradients = gradients(target, variables_to_change)
            change_list = []
            for index, variable in enumerate(variables_to_change):
                change_list.append(
                    assign(variable, variable - (self.learning_rate * variables_gradients[index])))
            return pack(change_list)


class nn(object):
    """ Supports neural network. """
    class SoftmaxOp(Op):
        def __call__(self, node_A, dim=-1, name=None):
            if name is None:
                name = "Softmax(%s, dim=%s)" % (node_A.name, dim)
            exp_node_A = exp(node_A)
            new_node = exp_node_A / \
                broadcastto_op(reduce_sum(exp_node_A, axis=dim), exp_node_A)
            new_node.name = name
            return new_node

    softmax = SoftmaxOp()
    relu = relu

    class SoftmaxCrossEntropyWithLogitsOp(Op):
        def __call__(self, logits, labels):
            return softmax_cross_entropy_op(logits, labels)

    softmax_cross_entropy_with_logits = SoftmaxCrossEntropyWithLogitsOp()
