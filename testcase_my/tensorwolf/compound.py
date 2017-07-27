""" Combine some operators together."""
from tensorwolf.ops import *


class nn(object):
    """ Supports neural network. """
    class SoftmaxOp(Op):
        def __call__(self, node_A, dim=-1, name=None):
            if name is None:
                name = "Softmax(%s, dim=%s)" % (node_A.name, dim)
            exp_node_A = exp(node_A)
            new_node = exp_node_A / \
                reduce_sum(exp_node_A, axis=dim, keep_dims=True)
            new_node.name = name
            return new_node

    softmax = SoftmaxOp()
