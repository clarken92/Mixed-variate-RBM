import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from collections import OrderedDict

from architecture.abstract.model import Model
from architecture.abstract.data_type import InputType
from architecture.abstract.regularizer import l1_grad, l2_grad
from utils.general_utils import init_weight, init_bias
from utils.theano_utils import store_grads_in_update


class RBM(Model):
    def __init__(self, v_dim=784, h_dim=500, input_type=InputType.binary,
                 W=None, b_h=None, b_v=None, sigma=None, input_var=None,
                 mrng=None, rng=None, name='', **kwargs):

        name = 'rbm' if name == '' else name
        super(RBM, self).__init__(name=name)

        model_file = kwargs.get('model_file')
        if model_file is not None:
            self.load(model_file)
            self._load_params()

        else:
            # v_dim is the dimensions of visible variable v. v_dim = D
            self.v_dim = v_dim
            # v_dim is the dimensions of visible variable v. h_dim = H
            self.h_dim = h_dim
            self.input_type = input_type

            seed = np.random.randint(1, 2**30)
            self._rng = RandomStreams(seed) if rng is None else rng
            self._mrng = MRG_RandomStreams(seed) if mrng is None else mrng

            self._build_params(W, b_h, b_v, sigma)

        self.input = input_var if input_var is not None else T.matrix('input')
        if self.input_type == InputType.poisson or self.input_type == InputType.replicated_softmax:
            self.total_count = T.sum(self.input, axis=1, keepdims=True)

    def _load_params(self):
        [self.W, self.b_h, self.b_v] = self.params

    def _build_params(self, W, b_h, b_v, sigma):
        self.params = []

        self.W = W if W is not None else init_weight(self.v_dim, self.h_dim, name=self.name+'-W')
        self.b_h = b_h if b_h is not None else init_bias(self.h_dim, name=self.name+'-b_h')
        self.b_v = b_v if b_v is not None else init_bias(self.v_dim, name=self.name+'-b_v')

        # sigma_v is not considered to be a param
        self.params.extend([self.W, self.b_h, self.b_v])

        # Truong hop gaussian co them sigma
        self.sigma_v = None
        if self.input_type == InputType.gaussian:
            print "Your input must be whitened to achieve the desire result."

            if sigma is not None:
                sigma = np.asarray(sigma)
                if sigma.ndim == 0:
                    print "Sigma is set to {} for all input dimensions.".format(sigma)
                    self.sigma_v = theano.shared(sigma * np.ones((self.v_dim, ), dtype=theano.config.floatX))
                    self.sigma_v.name = self.name + "-sigma_v"
                else:
                    assert sigma.ndim == 1 and sigma.shape[0] == self.v_dim, \
                        "Sigma must be 1D array with the length of {}".format(self.v_dim)
                    self.sigma_v = theano.shared(sigma)
                    self.sigma_v.name = self.name + "-sigma_v"
            else:
                print "Default value of sigma is 1.0 for all input dimensions."
                self.sigma_v = theano.shared(np.ones(self.v_dim, dtype=theano.config.floatX))
                self.sigma_v.name = self.name + "-sigma_v"

    def print_model_info(self):
        print "\nInfo of model {}".format(self.name)
        print "v_dims: {} | h_dim: {} | input_type: {}".format(self.v_dim, self.h_dim, self.input_type)

    def get_save(self):
        return [self.name, self.v_dim, self.h_dim, self.input_type,
                self._mrng, self._rng, self.params, self.sigma_v]

    def set_load(self, saved_data):
        [self.name, self.v_dim, self.h_dim, self.input_type,
         self._mrng, self._rng, self.params, self.sigma_v] = saved_data

    def score(self, v_data):
        free_fn = theano.function([self.input], self.free_energy(self.input))
        return free_fn(v_data)

    def reconstruct(self, v_data):
        h = self.h_given_v(self.input)
        rv = self.v_given_h(h)
        rec_fn = theano.function([self.input], rv)
        return rec_fn(v_data)

    def reconstruct_from_hidden(self, h_data):
        h = self.input.type('hidden')
        rv = self.v_given_h(h)
        rec_fn = theano.function([h], rv)
        return rec_fn(h_data)

    def encode(self, v_data):
        h_code = self.h_given_v(self.input)
        fn = theano.function([self.input], h_code)
        return fn(v_data)

    # Energy from many v an 1 h
    def energy(self, v, h):
        v_free = self.v_free_term(v)
        v_bias = self.v_bias_term(v)
        v_weight = self.v_weight_term(v)

        return -(v_free + v_bias + v_weight * h + T.dot(h, self.b_h))

    def free_energy(self, v):
        v_free = self.v_free_term(v)
        v_bias = self.v_bias_term(v)
        v_weight = self.v_weight_term(v)

        h_term = T.sum(T.log(1 + T.exp(v_weight + self.b_h)), axis=1)
        return -(v_bias + v_free + h_term)

    def v_weight_term(self, v):
        if self.input_type == InputType.gaussian:
            return T.dot(v/(self.sigma_v ** 2), self.W)
        else:
            return T.dot(v, self.W)

    def v_bias_term(self, v):
        # Note that for gaussian case, the v_bias should be negative
        if self.input_type == InputType.gaussian:
            return -T.sum((v - self.b_v) ** 2 / (2 * self.sigma_v ** 2), axis=1)
        else:
            return T.dot(v, self.b_v)

    def v_free_term(self, v):
        if self.input_type == InputType.poisson:
            return -T.sum(T.gammaln(1 + v), axis=1)
        else:
            return 0

    def rv(self, v):
        h = self.h_given_v(v)
        rv = self.v_given_h(h)
        return rv

    def h_given_v(self, v):
        v_weight = self.v_weight_term(v)
        p_h_v = T.nnet.sigmoid(v_weight + self.b_h)
        return p_h_v

    def v_given_h(self, h):
        if self.input_type == InputType.binary:
            p_v_h = T.nnet.sigmoid(self.b_v + T.dot(h, self.W.T))
            return p_v_h

        elif self.input_type == InputType.gaussian:
            mu_v = self.b_v + T.dot(h, self.W.T)
            return mu_v

        elif self.input_type == InputType.categorical or \
             self.input_type == InputType.replicated_softmax:
            p_v_h = T.nnet.softmax(self.b_v + T.dot(h, self.W.T))
            return p_v_h

        elif self.input_type == InputType.poisson:
            if not hasattr(self, 'total_count') or self.total_count is None:
                raise ValueError('Total count should be set for constrained Poisson')

            unconstrained_lmbd_v = T.exp(self.b_v + T.dot(h, self.W.T))
            lmbd_v = unconstrained_lmbd_v * 1.0 / T.sum(unconstrained_lmbd_v, axis=1, keepdims=True) \
                     * self.total_count
            return lmbd_v

    def sample_h_given_v(self, v0_sample):
        h1_mean = self.h_given_v(v0_sample)
        h1_sample = self._mrng.binomial(size=h1_mean.shape, n=1, p=h1_mean,
                                        dtype=theano.config.floatX)
        return [h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        if self.input_type == InputType.binary:
            v1_mean = self.v_given_h(h0_sample)
            v1_sample = self._mrng.binomial(size=v1_mean.shape, n=1, p=v1_mean,
                                           dtype=theano.config.floatX)
            return [v1_mean, v1_sample]

        elif self.input_type == InputType.gaussian:
            mu_v1 = self.v_given_h(h0_sample)  # Note that mu_v1 is returned

            v1_sample = self._mrng.normal(size=mu_v1.shape, avg=mu_v1, std=self.sigma_v,
                                          dtype=theano.config.floatX)
            return [mu_v1, v1_sample]
        # Note that there is constraint in the case of Multinomial

        elif self.input_type == InputType.categorical:
            prob_v1 = self.v_given_h(h0_sample)
            # Multinomial with n=1 (It is equal to categorical)
            v1_sample = self._mrng.multinomial(pvals=prob_v1, n=1, dtype=theano.config.floatX)
            return [prob_v1, v1_sample]

        elif self.input_type == InputType.poisson:
            lmbd_v1 = self.v_given_h(h0_sample)
            # We have to use RandomStreams, not MRG_RandomStreams
            v1_sample = self._rng.poisson(size=lmbd_v1.shape, lam=lmbd_v1,
                                          dtype=theano.config.floatX)
            return [lmbd_v1, v1_sample]

        elif self.input_type == InputType.replicated_softmax:
            if not hasattr(self, 'total_count') or self.total_count is None:
                raise ValueError('Total count should be set for replicated Softmax')

            prob_v1 = self.v_given_h(h0_sample)
            # We have to sample the vocabulary distribution given topic D times and sum over D samples
            v1_sample = self._mrng.multinomial(pvals=prob_v1, n=self.total_count, ndim=prob_v1.shape[1])
            return [prob_v1, v1_sample]

    # One step of gibbs sampling
    def gibbs_hvh(self, h0_sample):
        # Here we use v1_stat to show that it is sufficient statistics of v1
        [v1_stat, v1_sample] = self.sample_v_given_h(h0_sample)
        [h1_mean, h1_sample] = self.sample_h_given_v(v1_sample)

        return [v1_stat, v1_sample, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        [h1_mean, h1_sample] = self.sample_h_given_v(v0_sample)
        [v1_stat, v1_sample] = self.sample_v_given_h(h1_sample)

        return [h1_mean, h1_sample, v1_stat, v1_sample]

    def run_CD_from_h(self, k, data_h):
        start_h = T.matrix("start_h")
        # [v_stats, v_samples, h_means, h_samples], updates \
        outputs, updates \
            = theano.scan(fn=self.gibbs_hvh, outputs_info=[None, None, None, start_h],
                          n_steps=k, name="gibbs_hvh")
        # Return the last h_sample after k steps
        CD_fn = theano.function([start_h], outputs=outputs[-1], updates=updates)
        return CD_fn(data_h)

    def run_CD_from_v(self, k, data_v):
        start_v = T.matrix("start_v")
        # [h_means, h_samples, v_stats, v_samples], updates \
        outputs, updates \
            = theano.scan(fn=self.gibbs_vhv, outputs_info=[None, None, None, start_v],
                          n_steps=k, name="gibbs_vhv")
        # Return the last v_sample after k steps
        CD_fn = theano.function([start_v], outputs=outputs[-1], updates=updates)
        return CD_fn(data_v)

    # Return visible variables
    def _gibbs_vhv_to_v_fn(self, steps, persis_v, is_sample=True, name=''):
        [h_means, h_samples, v_stats, v_samples], updates \
            = theano.scan(self.gibbs_vhv,
                          outputs_info=[None, None, None, persis_v],
                          n_steps=steps,  # init_gibbs dung de init
                          name='gibbs_vhv')
        updates.update({persis_v: v_samples[-1]})
        if is_sample:
            gibbs_fn = theano.function([], v_samples[-1], updates=updates, name=name)
        else:
            gibbs_fn = theano.function([], v_stats[-1], updates=updates, name=name)
        return gibbs_fn

    # Also return visible variables
    def _gibbs_hvh_to_v_fn(self, steps, persis_h, is_sample=True, name=''):
        [v_stats, v_samples, h_means, h_samples], updates \
            = theano.scan(self.gibbs_hvh,
                          outputs_info=[None, None, None, persis_h],
                          n_steps=steps,  # init_gibbs dung de init
                          name='gibbs_hvh')
        updates.update({persis_h: h_samples[-1]})
        if is_sample:
            gibbs_fn = theano.function([], v_samples[-1], updates=updates, name=name)
        else:
            gibbs_fn = theano.function([], v_stats[-1], updates=updates, name=name)
        return gibbs_fn

    def sample_given_data(self, v_data, init_gibbs=1000, betw_gibbs=100, loops=10, is_sample=False):
        print "\nSample data from input using model {}".format(self.name)
        # Neu kich thuoc input la 1 thi phai chuyen no ve kich thuoc 2
        if len(v_data.shape) == 1:
            persis_v = theano.shared(np.asarray(v_data.reshape(1, v_data.shape[0]),
                                                dtype=theano.config.floatX))
        else:
            persis_v = theano.shared(np.asarray(v_data, dtype=theano.config.floatX))

        if init_gibbs > 0:
            init_sampling_fn = self._gibbs_vhv_to_v_fn(init_gibbs, persis_v,
                                                       is_sample=True, name='init_sampling_fn')
        else:
            init_sampling_fn = None

        sample_fn = self._gibbs_vhv_to_v_fn(betw_gibbs, persis_v,
                                            is_sample=is_sample, name='sample_fn')

        rvs_data = []
        if init_sampling_fn is not None:
            init_sampling_fn()
        for idx in range(loops):
            print "Running sampling loop %d" % idx
            rv_data = sample_fn()
            rvs_data.append(rv_data)

        return np.asarray(rvs_data)

    # Sample randomly
    # We start from h and run gibbs chain until it reaches equilibrium
    def sample(self, init_gibbs=1000, betw_gibbs=100, n_samples=20, loops=10, is_sample=False):
        print "\nSample random data using model {}".format(self.name)
        persis_h = theano.shared(np.zeros((n_samples, self.h_dim), dtype=theano.config.floatX))

        if init_gibbs > 0:
            init_sampling_fn = self._gibbs_hvh_to_v_fn(init_gibbs, persis_h,
                                                       is_sample=True, name='init_sampling_fn')
        else:
            init_sampling_fn = None
        sample_fn = self._gibbs_hvh_to_v_fn(betw_gibbs, persis_h,
                                            is_sample=is_sample, name='sample_fn')

        rvs_data = []
        if init_sampling_fn is not None:
            init_sampling_fn()
        for idx in range(loops):
            print "Running sampling loop %d" % idx
            rv_data = sample_fn()
            rvs_data.append(rv_data)

        return np.asarray(rvs_data)

    def get_cost_udpates(self, lr, k, persis_h, l1, l2, stable_update, store_grad):
        # Run one sample step to get h
        h_mean, h_sample = self.sample_h_given_v(self.input)

        # Run normal CD
        start_h = persis_h if persis_h is not None else h_sample

        [v_stats, v_samples, h_means, h_samples], updates \
            = theano.scan(fn=self.gibbs_hvh, outputs_info=[None, None, None, start_h],
                          n_steps=k, name="gibbs_hvh")

        vk = v_samples[-1]
        v_stat_k = v_stats[-1]

        if persis_h is not None:
            updates[persis_h] = h_samples[-1]

        cost = self.get_viewed_cost(self.input, v_stat_k)
        cost = T.mean(cost)

        # For stable update, use mean value instead of random sampled value
        if stable_update:
            print "\nStable update is set to be True"
            updates = self.params_updates(self.input, v_stat_k, lr, l1, l2, updates, store_grad)
        else:
            print "\nStable update is set to be False"
            updates = self.params_updates(self.input, vk, lr, l1, l2, updates, store_grad)

        # return cost, updates
        return cost, updates

    def get_viewed_cost(self, v0, vk_stat):
        # Binary cross-entropy
        cost = 0
        if self.input_type == InputType.binary:
            clip_vk_stat = T.clip(vk_stat, np.float32(0.000001), np.float32(0.999999))
            cost = -T.sum(v0 * T.log(clip_vk_stat) + (1 - v0) * T.log(1 - clip_vk_stat), axis=1)

        # Sum square error
        elif self.input_type == InputType.gaussian:
            cost = T.sum((v0 - vk_stat) ** 2, axis=1)

        # Categorical cross-entropy
        elif self.input_type == InputType.categorical:
            clip_vk_stat = T.clip(vk_stat, np.float32(0.000001), np.float32(0.999999))
            cost = -T.sum(v0 * T.log(clip_vk_stat), axis=1)

        elif self.input_type == InputType.poisson:
            clip_vk_stat = T.clip(vk_stat, np.float32(0.000001), np.inf)
            cost = -T.sum(-vk_stat + v0 * T.log(clip_vk_stat) - T.gammaln(1 + v0), axis=1)

        if self.input_type == InputType.replicated_softmax:
            clip_vk_stat = T.clip(vk_stat, np.float32(0.000001), np.inf)
            cost = -T.sum((v0 / self.total_count) * T.log(clip_vk_stat), axis=1)

        return cost

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

    def nll_grad_formula(self, v0, vk):
        n_instances = v0.shape[0]

        h0 = self.h_given_v(v0)
        hk = self.h_given_v(vk)

        gW = (T.dot(vk.T, hk) - T.dot(v0.T, h0)) / n_instances
        gb_h = T.mean(hk - h0, axis=0)

        if self.input_type == InputType.gaussian:
            gb_v = T.mean((vk - v0) / (self.sigma_v ** 2), axis=0)

            ugz_v = (((vk - self.b_v) ** 2 - 2 * vk * T.dot(hk, self.W.T)) - \
                    ((v0 - self.b_v) ** 2 - 2 * v0 * T.dot(h0, self.W.T))) / (self.sigma_v ** 2)
            gz_v = T.mean(ugz_v, axis=0)

            grads = [gW, gb_h, gb_v, gz_v]

        else:
            gb_v = T.mean(vk - v0, axis=0)
            grads = [gW, gb_h, gb_v]

        return grads

    def nll_grad_theano(self, v0, vk):
        cost = T.mean(self.free_energy(v0)) - T.mean(self.free_energy(vk))
        # Note here we have to use consider_constant
        grads = T.grad(cost, self.params, consider_constant=[vk])
        return grads

    def grad_check(self, data_v0, data_vk):
        # data_v0 and data_vk is numpy array
        # data_vk is computed by calling CD-k
        v0 = T.matrix('v0')
        vk = T.matrix('vk')

        theano_grads = self.nll_grad_theano(v0, vk)
        formula_grads = self.nll_grad_formula(v0, vk)

        grad_diffs = []
        for t_grad, f_grad in zip(theano_grads, formula_grads):
            grad_diffs.append(abs(t_grad - f_grad))

        grad_test_fn = theano.function([v0, vk], grad_diffs)

        diffs_results = grad_test_fn(data_v0, data_vk)
        for i in xrange(len(self.params)):
            if self.params[i].name is not None:
                name = self.params[i].name
            else:
                name = ""
            print ("Max " + name + " diffs: {}").format(np.max(diffs_results[i]))
            print ("Min " + name + " diffs: {}").format(np.min(diffs_results[i]))
            print ("Average " + name + " diffs: {}").format(np.mean(diffs_results[i]))
            print
        return diffs_results

    def config_train(self, **kwargs):
        k = kwargs.get('CD_k')
        persis_h_data = kwargs.get('persis_h')
        l1 = kwargs.get('L1')
        l2 = kwargs.get('L2')

        if l1 is None:
            print "L1 should be set to enable sparse weight regularization"
        if l2 is None:
            print "L2 should be set to enable sparse weight regularization"

        stable_update = kwargs.get('stable_update')
        if stable_update is None:
            stable_update = False

        store_grad = kwargs.get('store_grad')
        if store_grad is None:
            store_grad = False

        self._build_train(k, persis_h_data, l1, l2, stable_update, store_grad)

    # persis_v_data is a numpy array
    def _build_train(self, k, persis_h_data, l1, l2, stable_update, store_grad):
        print "\nBuild training function of model {}".format(self.name)

        if persis_h_data is not None:
            persis_h = theano.shared(persis_h_data, borrow=True)
        else:
            persis_h = None

        lr = T.scalar('lr')
        cost, updates = self.get_cost_udpates(lr, k, persis_h, l1, l2, stable_update, store_grad)
        print "\nBuild computation graph for training function of model {}".format(self.name)
        self.train_fn = theano.function([self.input, lr], cost, updates=updates)

        rv = self.v_given_h(self.h_given_v(self.input))
        test_cost = self.get_viewed_cost(self.input, rv)
        test_cost = T.mean(test_cost)
        print "\nBuild computation graph for validation function of model {}".format(self.name)
        self.valid_fn = theano.function([self.input], test_cost)