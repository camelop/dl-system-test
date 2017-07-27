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
