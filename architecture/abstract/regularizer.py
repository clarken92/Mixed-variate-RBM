import theano.tensor as T


def l1_cost(W, l1):
    return l1 * T.sum(abs(W))


def l1_grad(W, l1):
    l1_reg = l1_cost(W, l1)
    gW = T.grad(l1_reg, W)
    return gW


def l2_cost(W, l2):
    return l2 * T.sum(W ** 2)


def l2_grad(W, l2):
    l2_reg = l2_cost(W, l2)
    gW = T.grad(l2_reg, W)
    return gW