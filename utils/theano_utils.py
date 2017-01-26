import numpy as np
import theano
from collections import OrderedDict

def store_grads_in_update(params, grads, updates):
    shared_grads = []
    param_shapes = [param.get_value(borrow=True).shape for param in params]
    if updates is None:
        updates = OrderedDict()

    for i, param in enumerate(params):
        if not isinstance(grads[i], (int, float, bool)):
            shared_grad = theano.shared(np.zeros(param_shapes[i], dtype=param.dtype),
                                        name=param.name + '-grad')

            updates[shared_grad] = grads[i].astype(param.dtype)
            shared_grads.append(shared_grad)

    return shared_grads, updates
