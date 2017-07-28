""" define the behaviors of nodes """
from __future__ import absolute_import
import numpy as np
#reference: dlsys-autodiff
variables = {}


class Node(object):
    """Node in a computation graph."""

    def __init__(self):
        """Constructor, new node is indirectly created by Op object call method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object,
                e.g. add_op if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant.
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            new_node = add_op(self, constant(other))
        return new_node

    def __mul__(self, other):
        """Multiplying two nodes return a new node."""
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mul_op(self, constant(other))
        return new_node

    # Allow left-hand-size add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __sub__(self, rhs):
        """node_self - node_rhs, return a new node."""
        if isinstance(rhs, Node):
            new_node = sub_op(self, rhs)
        else:
            new_node = sub_op(self, constant(rhs))
        return new_node

    def __rsub__(self, lhs):
        """node_lhs - node_self, return a new node."""
        if isinstance(lhs, Node):
            new_node = sub_op(lhs, self)
        else:
            new_node = sub_op(constant(lhs), self)
        return new_node

    def __div__(self, rhs):
        """node_self / node_rhs, return a new node."""
        if isinstance(rhs, Node):
            new_node = div_op(self, rhs)
        else:
            new_node = div_op(self, constant(rhs))
        return new_node

    def __rdiv__(self, lhs):
        """node_lhs / node_self, return a new node."""
        if isinstance(lhs, Node):
            new_node = div_op(lhs, self)
        else:
            new_node = div_op(constant(lhs), self)
        return new_node

    __floordiv__ = __div__
    __rfloordiv__ = __rdiv__

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __neg__(self):
        """-node_self, return a new node."""
        new_node = constant(0) - self
        return new_node

    def __str__(self):
        """Allow print to display node name."""
        return self.name


class Op(object):
    """Op represents operations performed on nodes."""

    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given output gradient, compute partial gradient to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError


class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        output_val = input_vals[0] + input_vals[1]
        return output_val

    def gradient(self, node, output_grad):
        return [adapt(output_grad, node.inputs[0]), adapt(output_grad, node.inputs[1])]


class SubOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s-%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        output_val = input_vals[0] - input_vals[1]
        return output_val

    def gradient(self, node, output_grad):
        return [adapt(output_grad, node.inputs[0]),
                adapt(constant(0.) - output_grad, node.inputs[1])]


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        output_val = input_vals[0] * input_vals[1]
        return output_val

    def gradient(self, node, output_grad):
        return [adapt(node.inputs[1] * output_grad, node.inputs[0]),
                adapt(node.inputs[0] * output_grad, node.inputs[1])]


class DivOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s/%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        output_val = input_vals[0] / input_vals[1]
        return output_val

    def gradient(self, node, output_grad):
        return [adapt(output_grad / node.inputs[1], node.inputs[0]),
                adapt(((output_grad * node.inputs[0] * constant(-1)) /
                       node.inputs[1]) / node.inputs[1], node.inputs[1])]


class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (
            node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals):
        if ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is False)):
            output_val = np.matmul(input_vals[0], input_vals[1])
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is False)):
            output_val = np.matmul(
                np.transpose(input_vals[0]), input_vals[1])
        elif ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is True)):
            output_val = np.matmul(
                input_vals[0], np.transpose(input_vals[1]))
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is True)):
            output_val = np.matmul(
                np.transpose(input_vals[0]), np.transpose(input_vals[1]))
        return output_val

    def gradient(self, node, output_grad):
        if ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            lhs_grad = matmul(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = matmul(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A^T dY
            lhs_grad = matmul(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = matmul(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B^T, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = matmul(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B^T)^T=B dY^T, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = matmul(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        return [adapt(lhs_grad, node.inputs[0]),
                adapt(rhs_grad, node.inputs[1])]


class PlaceholderOp(Op):
    def __call__(self, dtype, shape=None, name="Placeholder"):
        """Creates a placeholder node."""
        new_node = Op.__call__(self)
        new_node.const_attr = (shape, dtype)
        new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        assert False, "placeholder %s values provided by feed_dict" % (
            node.name)

    def gradient(self, node, output_grad):
        return None


class VariableOp(Op):
    def __call__(self, initial_value, dtype=None, shape=None, name="Variable"):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        # check the input's shape
        if shape is not None:
            assert shape == initial_value.shape
        # insert new node to global dict
        if dtype is not None:
            if isinstance(initial_value, np.ndarray):
                variables[new_node] = initial_value.astype(dtype)
            else:
                variables[new_node] = np.array(initial_value).astype(dtype)
        else:
            variables[new_node] = initial_value
        new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        if node.const_attr is None:
            raise UnboundLocalError
        return node.const_attr

    def gradient(self, node, output_grad):
        return None


class ConstantOp(Op):
    def __call__(self, initial_value, dtype=None, shape=None, name="Const"):
        """Creates a constant node."""
        new_node = Op.__call__(self)
        new_node.const_attr = np.array(
            initial_value).reshape(shape).astype(dtype)
        new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        return node.const_attr

    def gradient(self, node, output_grad):
        return None


class ZerosLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.zeroslike_op(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        output_val = np.zeros(input_vals[0].shape)
        return output_val

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class OnesLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.ones(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        output_val = np.ones(input_vals[0].shape)
        return output_val

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class ReduceSumOp(Op):
    def __call__(self, node_A, axis=None, keep_dims=False, reduction_indices=None):
        new_node = Op.__call__(self)
        if axis is None and reduction_indices is not None:
            axis = tuple(reduction_indices)
        new_node.inputs = [node_A]
        new_node.name = "ReduceSum(%s, axis=%s, keep_dims=%s)" % (
            node_A, axis, keep_dims)
        new_node.const_attr = (axis, keep_dims)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        output_val = np.sum(
            input_vals[0], axis=node.const_attr[0], keepdims=node.const_attr[1])
        return output_val

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0])]


class ReduceMeanOp(Op):
    def __call__(self, node_A, axis=None, keep_dims=False, reduction_indices=None):
        new_node = Op.__call__(self)
        if axis is None and reduction_indices is not None:
            axis = tuple(reduction_indices)
        new_node.inputs = [node_A]
        new_node.name = "ReduceMean(%s, axis=%s, keep_dims=%s)" % (
            node_A, axis, keep_dims)
        new_node.const_attr = (axis, keep_dims)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        output_val = np.mean(
            input_vals[0], axis=node.const_attr[0], keepdims=node.const_attr[1])
        # cross entropy
        # print(output_val)
        return output_val

    def gradient(self, node, output_grad):
        return [adapt(broadcastto_op(output_grad, node.inputs[0]) /
                      reduce_sum(oneslike_op(node.inputs[0]), axis=node.const_attr[0], keep_dims=node.const_attr[1]), node.inputs[0])]


class ReduceShapeSumOp(Op):
    def __call__(self, node_A, node_B):
        """Creates a node that represents sum(node_A) to shape node_B.shape"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReduceShapeSum(%s, %s.shape)" % (
            node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):

        assert len(input_vals) == 2

        output_val = input_vals[0]
        while len(output_val.shape) > len(input_vals[1].shape):
            output_val = np.sum(output_val, axis=0)
        for dim in range(len(output_val.shape)):
            if output_val.shape[dim] > input_vals[1].shape[dim]:
                assert input_vals[1].shape[dim] == 1
                output_val = np.sum(output_val, axis=dim, keepdims=True)

        return output_val

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0]), zeroslike_op(node.inputs[1])]


class AdaptShapeOp(Op):
    def __call__(self, node_A, node_B):
        """(Adapt the shape) Creates a node that represents sum(node_A) to shape node_B.shape
            for now it is the same as ReduceShapeSum"""
        new_node = reduceshapesum_op(node_A, node_B)
        new_node.name = "Adapt(%s, %s.shape)" % (
            node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        raise NotImplementedError

    def gradient(self, node, output_grad):
        raise NotImplementedError


class ReduceShapeMeanOp(Op):
    def __call__(self, node_A, node_B):
        """Creates a node that represents mean(node_A) to shape node_B.shape"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReduceShapeMean(%s, shape=%s.shape)" % (
            node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        output_val = input_vals[0]
        while len(output_val.shape) > len(input_vals[1].shape):
            output_val = np.mean(output_val, axis=0)
        for dim in range(len(output_val.shape)):
            if output_val.shape[dim] > input_vals[1].shape[dim]:
                assert input_vals[1].shape[dim] == 1
                output_val = np.mean(output_val, axis=dim, keepdims=True)
        return output_val

    def gradient(self, node, output_grad):
        raise NotImplementedError


class BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        """Creates a node that represents np.broadcast_to(node_A, node_B.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "BroadcastTo(%s,%s.shape)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        output_val = input_vals[0]
        # not complete yet
        if len(output_val.shape) < len(input_vals[1].shape):
            front_align = True
            for dim, in_size in enumerate(output_val.shape):
                if input_vals[1].shape[dim] != in_size:
                    front_align = False
                    break
            new_shape = output_val.shape
            if front_align:
                while len(new_shape) < len(input_vals[1].shape):
                    new_shape = new_shape + (1,)
            output_val.resize(new_shape)
        output_val = np.broadcast_to(output_val, input_vals[1].shape)
        return output_val

    def gradient(self, node, output_grad):
        grad_A = reduceshapesum_op(output_grad, node.inputs[0])
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]


class ExpOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.exp(node_A)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Exp(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        # print(input_vals)
        output_val = np.exp(input_vals[0])
        return output_val

    def gradient(self, node, output_grad):
        return [adapt(output_grad * exp(node.inputs[0]), node.inputs[0])]


class LogOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.log(node_A)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Log(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        output_val = np.log(input_vals[0])
        return output_val

    def gradient(self, node, output_grad):
        return [adapt(output_grad / node.inputs[0], node.inputs[0])]


class VariablesInitOp(Op):
    def __call__(self):
        """Feed the global 'variables' into the exact variables."""
        new_node = Op.__call__(self)
        new_node.inputs = []
        new_node.name = "Global_Variables_Initializer"
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 0
        for key, value in variables.items():
            key.const_attr = value
        return 0  # as the signal of success

    def gradient(self, node, output_grad):
        raise NotImplementedError


class AssignOp(Op):
    # notice: here, the definition of node_A is different from others
    def __call__(self, node_A, node_B):
        """Assign input[0] with the value of input[1], return input[1] after assignment"""
        new_node = Op.__call__(self)
        if not isinstance(node_B, Node):
            node_B = constant(node_B)
        new_node.inputs = [node_B]
        new_node.const_attr = node_A
        new_node.name = "(%s:=%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert isinstance(node.const_attr.op, VariableOp)
        node.const_attr.const_attr = input_vals[0]
        return input_vals[0]

    def gradient(self, node, output_grad):
        raise NotImplementedError


class EqualOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s==%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        output_val = np.equal(input_vals[0], input_vals[1])
        return output_val

    def gradient(self, node, output_grad):
        raise NotImplementedError


class ArgMaxOp(Op):
    def __call__(self, node_A, axis=None, name=None, dimension=None):
        # I don't know what dimension stands for...
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = axis
        if name is None:
            new_node.name = "Argmax(%s, axis=%s)" % (node_A.name, axis)
        else:
            new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        output_val = np.argmax(input_vals[0], axis=node.const_attr)
        return output_val

    def gradient(self, node, output_grad):
        raise NotImplementedError


class CastOp(Op):
    def __call__(self, node_A, dtype, name=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = dtype
        if name is None:
            new_node.name = "Cast(%s, dtype=%s)" % (node_A.name, dtype)
        else:
            new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        output_val = input_vals[0].astype(node.const_attr)
        return output_val

    def gradient(self, node, output_grad):
        raise NotImplementedError


class PackOp(Op):
    def __call__(self, node_list, name=None):
        new_node = Op.__call__(self)
        new_node.inputs = node_list
        if name is None:
            new_node.name = "Pack(%s)" % (
                str([node.name for node in node_list]))
        else:
            new_node.name = name
        return new_node

    def compute(self, node, input_vals):
        # nothing should be done since this is the top
        return None

    def gradient(self, node, output_grad):
        raise NotImplementedError


# Create global singletons of operators.
add_op = AddOp()
sub_op = SubOp()
mul_op = MulOp()
div_op = DivOp()
matmul = MatMulOp()
placeholder = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
reduce_sum = ReduceSumOp()
reduce_mean = ReduceMeanOp()
reduceshapesum_op = ReduceShapeSumOp()
reduceshapemean_op = ReduceShapeMeanOp()
broadcastto_op = BroadcastToOp()
global_variables_initializer = VariablesInitOp()
Variable = VariableOp()
constant = ConstantOp()
assign = AssignOp()
exp = ExpOp()
log = LogOp()
equal = EqualOp()
argmax = ArgMaxOp()
cast = CastOp()
adapt = AdaptShapeOp()
pack = PackOp()
