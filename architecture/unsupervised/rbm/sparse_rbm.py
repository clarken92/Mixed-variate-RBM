import numpy as np
import theano
from theano.ifelse import ifelse
import theano.tensor as T
from collections import OrderedDict

from rbm import RBM
from architecture.abstract.data_type import InputType
from architecture.abstract.regularizer import l1_grad, l2_grad
from utils.theano_utils import store_grads_in_update

# Implementation based on Hinton's paper: A Practical Guide to Training Restricted Boltzmann Machine
class SparseRBM(RBM):
    def __init__(self, v_dim=784, h_dim=500, input_type=InputType.binary,
                 W=None, b_h=None, b_v=None, sigma=None, input_var=None, mrng=None, rng=None, name='',
                 sparsity_coeff=1.0, sparsity_decay=0.9, sparsity_target=0.05, **kwargs):

        model_file = kwargs.get('model_file')
        if model_file is not None:
            super(SparseRBM, self).__init__(model_file=model_file)

        else:
            name = 'sparse_rbm' if name == '' else name
            self.sparsity_coeff = sparsity_coeff
            self.sparsity_decay = sparsity_decay
            self.sparsity_target = sparsity_target

            #self.is_first = theano.shared(np.array(True, dtype=np.int32), name='is_first')
            self.act_prob = theano.shared(0.5 * np.ones(h_dim, dtype=theano.config.floatX),
                                          name='act_prob') # Prior belief

            super(SparseRBM, self).__init__(v_dim=v_dim, h_dim=h_dim, input_type=input_type,
                                            W=W, b_h=b_h, b_v=b_v, sigma=sigma, input_var=input_var,
                                            mrng=mrng, rng=rng, name=name, **kwargs)

    def get_save(self):
        return [self.name, self.v_dim, self.h_dim, self.input_type,
                self._mrng, self.params, self.sigma_v,
                self.sparsity_coeff, self.sparsity_decay, self.sparsity_target, self.act_prob]

    def set_load(self, saved_data):
        [self.name, self.v_dim, self.h_dim, self.input_type,
         self._mrng, self.params, self.sigma_v,
         self.sparsity_coeff, self.sparsity_decay, self.sparsity_target, self.act_prob] = saved_data

    def _average_hidden_activation(self, v, updates):
        h = self.h_given_v(v)  # h has shape (n, H)
        mean_h = T.mean(h, axis=0)  # mean activation on current mini-batch

        # Update activation
        new_act_prob = self.sparsity_decay * self.act_prob + (1 - self.sparsity_decay) * mean_h
        #new_act_prob = ifelse(self.is_first, mean_h, new_act_prob)

        updates[self.act_prob] = new_act_prob # act_prob has the same shape as h
        #updates[self.is_first] = theano.tensor.TensorConstant(self.is_first.type, False, name='is_first')

        return new_act_prob, updates

    def _sparsity_reg(self, v, updates):
        new_act_prob, updates = self._average_hidden_activation(v, updates)

        # sum here is not the sum over mini-batch instances
        # activation over mini-batch instances was averaged out
        # sum here is just the sum over all units of average hidden activation
        # new_act_prob has shape (1, H)
        new_act_prob = T.clip(new_act_prob, np.float32(0.000001), np.float32(0.999999))

        sparsity_reg = self.sparsity_coeff * T.sum(
            T.nnet.binary_crossentropy(new_act_prob, self.sparsity_target))
        return sparsity_reg, updates

    def _sparsity_grads_theano(self, v, updates):
        sparsity_reg, updates = self._sparsity_reg(v, updates)
        [gW, gb_h] = T.grad(sparsity_reg, [self.W, self.b_h])
        grads = [gW, gb_h, 0]

        return grads, updates

    # It just a way to work around
    # We should find the true formula for gb_h and gW
    def _sparsity_grads_formula(self, v, updates):
        new_act_prob, updates = self._average_hidden_activation(v, updates)
        gb_h = self.sparsity_coeff * (new_act_prob - self.sparsity_target)
        grads = [0, gb_h, 0]

        return grads, updates

    def params_updates(self, v0, vk, lr, l1, l2, updates, store_grad):
        if updates is None:
            updates = OrderedDict()
        if store_grad:
            self.stored_grads = OrderedDict()

        grads = [0 for _ in xrange(len(self.params))]

        o_grads = self.nll_grad_formula(v0, vk)
        grads = [grads[i] + o_grads[i] for i in xrange(len(self.params))]

        if store_grad:
            print "\nGradients over negative log-likelihood are stored in original_grads"
            o_shared_grads, updates = store_grads_in_update(self.params, o_grads, updates)
            self.stored_grads['original_grads'] = o_shared_grads

        # Parameters update to enforce sparsity among hidden units
        print "Add hidden sparsity regularization with decay ({}) to " \
              "parameter updates".format(self.sparsity_decay)
        # We must use updates here
        # Gradient of Max Pooling is not very good
        s_grads, updates = self._sparsity_grads_theano(v0, updates)
        #s_grads, updates = self._sparsity_grads_formula(v0, updates)
        grads = [grads[i] + s_grads[i] for i in xrange(len(self.params))]

        if store_grad:
            print "\nGradients over sparsity regularization are stored in sparsity_grads"
            s_shared_grads, updates = store_grads_in_update(self.params, s_grads, updates)
            self.stored_grads['sparsity_grads'] = s_shared_grads

        if l1 is not None:
            print "Add L1 regularization ({}) to parameter updates".format(l1)
            l1_gW = l1_grad(self.W, l1)
            grads[0] = grads[0] + l1_gW

            if store_grad:
                print "\nGradients over L1 regularization are stored in l1_grads"
                l1_shared_grads, updates = store_grads_in_update([self.W], [l1_gW], updates)
                self.stored_grads['l1_grads'] = l1_shared_grads

        if l2 is not None:
            print "Add L2 regularization ({}) to parameter updates".format(l2)
            l2_gW = l2_grad(self.W, l2)
            grads[0] = grads[0] + l2_gW

            if store_grad:
                print "\nGradients over L2 regularization are stored in l2_grads"
                l2_shared_grads, updates = store_grads_in_update([self.W], [l2_gW], updates)
                self.stored_grads['l2_grads'] = l2_shared_grads

        if store_grad:
            print "\nGradients over total cost are stored in total_grads"
            t_shared_grads, updates = store_grads_in_update(self.params, grads, updates)
            self.stored_grads['total_grads'] = t_shared_grads

        grads = [grad.astype(theano.config.floatX) for grad in grads]

        if self.check_learning_algor():
            params_updates = self.learning_algor(grads, self.params, lr, **self.learning_config)
            updates.update(params_updates)
        else:
            print "\nSimple SGD is used as training algorithm"
            for grad, param in zip(grads, self.params):
                updates[param] = param - grad * lr

        return updates
