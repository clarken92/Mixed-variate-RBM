import numpy as np
import scipy.sparse as sp
import theano
from theano import sparse
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

from collections import OrderedDict

from utils.general_utils import init_weight, init_bias
from utils.theano_utils import store_grads_in_update
from architecture.abstract.regularizer import l1_grad, l2_grad
from architecture.abstract.model import Model
from architecture.abstract.data_type import InputType

class VisibleLayer(object):
    def __init__(self, v_dim, h_dim, v_type, mrng=None, rng=None, name=''):

        self.name = name if name != '' else 'v_layer'

        self.v_dim = v_dim
        self.h_dim = h_dim
        self.v_type = v_type

        seed = np.random.randint(1, 2 ** 30)
        self._rng = RandomStreams(seed) if rng is None else rng
        self._mrng = MRG_RandomStreams(seed) if mrng is None else mrng

        self._build_params()

    def set_total_count(self, total_count):
        if not (self.v_type == InputType.poisson):
            raise ValueError("The input type should be Poisson to set total count")
        self.total_count = total_count

    def _build_params(self):
        # W to connect with hidden layer
        self.params = []
        if self.v_type == InputType.poisson:
            init_W = np.random.uniform(low=-1/self.h_dim,
                                       high=1/self.h_dim,
                                       size=(self.v_dim, self.h_dim))
            self.W = init_weight(self.v_dim, self.h_dim, value=init_W, name=self.name + '-W')
        else:
            self.W = init_weight(self.v_dim, self.h_dim, name=self.name + '-W')
        self.b_v = init_bias(self.v_dim, name=self.name + '-b_v')

        # Ca binary, gaussian, and categorical
        self.params.extend([self.W, self.b_v])

        # Truong hop gaussian co them sigma
        if self.v_type == InputType.gaussian:
            self.sigma_v = T.ones(shape=(self.v_dim, ), dtype=theano.config.floatX)
            self.sigma_v.name = self.name + "-sigma_v"

    # Result in a vector of (n, 1)
    def v_free_term(self, v):
        if self.v_type == InputType.poisson:
            return -T.sum(T.gammaln(1+v), axis=1)
        else:
            return 0

    # Result in a vector of (n, 1)
    def v_bias_term(self, v):
        # Note that for gaussian case, the v_bias should be negative
        if self.v_type == InputType.gaussian:
            return -T.sum((v - self.b_v) ** 2 / (2 * self.sigma_v ** 2), axis=1)
        else:
            return T.dot(v, self.b_v)

    # Result in a vector of (n, H)
    def v_weight_term(self, v):
        if self.v_type == InputType.gaussian:
            return T.dot((v /(self.sigma_v ** 2)), self.W)
        else:
            return T.dot(v, self.W)

    # Only support binary, gaussian and categorical
    def v_given_h(self, h):
        if self.v_type == InputType.binary:
            p_v_h = T.nnet.sigmoid(self.b_v + T.dot(h, self.W.T))
            return p_v_h

        elif self.v_type == InputType.gaussian:
            mu_v = self.b_v + T.dot(h, self.W.T)
            return mu_v

        elif self.v_type == InputType.categorical:
            p_v_h = T.nnet.softmax(self.b_v + T.dot(h, self.W.T))
            return p_v_h

        elif self.v_type == InputType.poisson:
            if not hasattr(self, 'total_count') or self.total_count is None:
                raise ValueError('Total count should be set for constrained Poisson')

            unconstrained_lmbd_v = T.exp(self.b_v + T.dot(h, self.W.T))
            lmbd_v = unconstrained_lmbd_v * 1.0 / T.sum(unconstrained_lmbd_v, axis=1, keepdims=True) \
                     * self.total_count
            return lmbd_v

    # Only support binary, gaussian and categorical
    def sample_v_given_h(self, h0_sample):
        if self.v_type == InputType.binary:
            v1_mean = self.v_given_h(h0_sample)
            v1_sample = self._mrng.binomial(size=v1_mean.shape, n=1, p=v1_mean,
                                            dtype=theano.config.floatX)
            return [v1_mean, v1_sample]

        elif self.v_type == InputType.gaussian:
            mu_v1 = self.v_given_h(h0_sample)  # Note that mu_v1 is returned

            v1_sample = self._mrng.normal(size=mu_v1.shape, avg=mu_v1, std=self.sigma_v,
                                          dtype=theano.config.floatX)
            return [mu_v1, v1_sample]

        # Note that there is constraint in the case of Multinomial
        elif self.v_type == InputType.categorical:
            prob_v1 = self.v_given_h(h0_sample)
            v1_sample = self._mrng.multinomial(pvals=prob_v1, n=1, dtype=theano.config.floatX)
            return [prob_v1, v1_sample]

        elif self.v_type == InputType.poisson:
            lmbd_v1 = self.v_given_h(h0_sample)
            # We have to use RandomStreams, not MRG_RandomStreams
            v1_sample = self._rng.poisson(size=lmbd_v1.shape, lam=lmbd_v1,
                                          dtype=theano.config.floatX)
            return [lmbd_v1, v1_sample]

    def l1_grad(self, l1):
        gW = l1_grad(self.W, l1)
        return [gW, 0]

    def l2_grad(self, l2):
        gW = l2_grad(self.W, l2)
        return [gW, 0]

    def nll_grad_formula(self, v0, vk, h0, hk):
        n_instances = v0.shape[0]

        gW = (T.dot(vk.T, hk) - T.dot(v0.T, h0)) / n_instances

        if self.v_type == InputType.gaussian:
            gb_v = T.mean((vk - v0) / (self.sigma_v ** 2), axis=0)
            grads = [gW, gb_v]
        else:
            gb_v = T.mean(vk - v0, axis=0)
            grads = [gW, gb_v]

        return grads

    def get_viewed_cost(self, v0, vk_stat):
        # Binary cross-entropy
        cost = 0
        if self.v_type == InputType.binary:
            # Clip to avoid log(0)
            clip_vk_stat = T.clip(vk_stat, np.float32(0.000001), np.float32(0.999999))
            cost = -T.sum(v0 * T.log(clip_vk_stat) + (1 - v0) * T.log(1 - clip_vk_stat), axis=1)

        # Sum square error
        elif self.v_type == InputType.gaussian:
            cost = T.sum((v0 - vk_stat) ** 2, axis=1)

        # Categorical cross-entropy
        elif self.v_type == InputType.categorical:
            clip_vk_stat = T.clip(vk_stat, np.float32(0.000001), np.float32(0.999999))
            cost = -T.sum(v0 * T.log(clip_vk_stat), axis=1)

        elif self.v_type == InputType.poisson:
            clip_vk_stat = T.clip(vk_stat, np.float32(0.000001), np.inf)
            cost = -T.sum(-vk_stat + v0 * T.log(clip_vk_stat) - T.gammaln(1 + v0), axis=1)

        return cost

    def get_params(self):
        return self.params


class MixedRBM(Model):
    def __init__(self, v_dim=784, h_dim=500, v_types=[], v_indices=[],
                 b_h=None, input_var=None, mrng=None, rng=None, name='', **kwargs):
        name = 'mixed_rbm' if name == '' else name

        super(MixedRBM, self).__init__(name,)
        self.input = T.matrix('input')
        self.n_instances = self.input.shape[0]

        model_file = kwargs.get('model_file')
        if model_file is not None:
            self.load(model_file)
            self._load_params()

        else:
            self.v_dim = v_dim
            self.h_dim = h_dim
            self.v_types = v_types
            self.v_indices = v_indices

            seed = np.random.randint(1, 2 ** 30)
            self._mrng = MRG_RandomStreams(seed) if mrng is None else mrng
            self._rng = RandomStreams(seed) if rng is None else rng
            self._rng = None

            if hasattr(self.v_indices[0], '__iter__'):
                self.v_ranges = self.v_indices
            else:
                self.v_ranges = [None] * len(self.v_indices)
                for i in xrange(len(self.v_indices)):
                    self.v_ranges[i] = range(self.v_indices[i], self.v_indices[i + 1]) \
                        if i < len(self.v_indices) - 1 else range(self.v_indices[i], v_dim)

            self.v_layers = []

            for i in xrange(len(self.v_ranges)):
                self.v_ranges[i] = np.asarray(self.v_ranges[i], dtype=np.int32)
                v_layer = VisibleLayer(v_dim=len(self.v_ranges[i]),
                                       h_dim=self.h_dim,
                                       v_type=self.v_types[i],
                                       name='v_layer({})'.format(i),
                                       mrng=self._mrng, rng=self._rng)
                if v_types[i] == InputType.poisson:
                    total_count = T.sum(self.input[:, self.v_ranges[i]], axis=1, keepdims=True)
                    v_layer.set_total_count(total_count)

                self.v_layers.append(v_layer)

            self._build_mask()
            self._build_params()

    def print_model_info(self):
        print "\nInfo of model {}".format(self.name)
        print "v_dims: {} | h_dim: {}".format(self.v_dim, self.h_dim)
        for i in xrange(len(self.v_types)):
            print "v_types: {} | v_ranges: {}".format(self.v_types[i], self.v_ranges[i])

    def get_save(self):
        return [self.name, self.v_dim, self.h_dim, self.v_indices, self.v_types,
                self._mrng, self._rng, self.big_mask,
                self.v_ranges, self.v_layers, self.b_h]

    def set_load(self, saved_data):
        [self.name, self.v_dim, self.h_dim, self.v_indices, self.v_types,
         self._mrng, self._rng, self.big_mask,
         self.v_ranges, self.v_layers, self.b_h] = saved_data

    def _load_params(self):
        self.params = [self.b_h]
        for i in xrange(len(self.v_layers)):
            self.params.extend(self.v_layers[i].get_params())

    def _build_params(self):
        self.b_h = init_bias(dim=self.h_dim, name=self.name + '-b_h')
        self.params = [self.b_h]

        for i in xrange(len(self.v_layers)):
            self.params.extend(self.v_layers[i].get_params())

    def _build_mask(self):
        big_m = np.zeros((self.v_dim, self.v_dim), dtype=theano.config.floatX)
        k = 0
        for i in xrange(len(self.v_ranges)):
            for j in xrange(len(self.v_ranges[i])):
                big_m[k, self.v_ranges[i][j]] = 1
                k += 1
        # self.big_mask = theano.shared(big_m, name='big_mask')
        # Sparse mask
        self.big_mask = sparse.shared(sp.csc_matrix(big_m), name='big_mask')

    def encode(self, v_data):
        h_code = self.h_given_v(self.input)
        fn = theano.function([self.input], h_code)
        return fn(v_data)

    def get_weight(self):
        Ws = [v_layer.W for v_layer in self.v_layers]
        return sparse.structured_dot(self.big_mask.T, T.concatenate(Ws, axis=0))

    def _vs(self, v):
        # Mac loi ngo ngan o day ma mo mai khong ra
        # return [v[v_range] for v_range in self.v_ranges]
        return [v[:, v_range] for v_range in self.v_ranges]

    def score(self, data):
        free_fn = theano.function([self.input], self.free_energy(self.input))
        return free_fn(data)

    # Energy from many v an 1 h
    def energy(self, vs, h):
        v_free = 0
        v_bias = 0
        v_weight = 0
        for i in xrange(len(vs)):
            v_free += self.v_layers[i].v_free_term(vs[i])
            v_bias += self.v_layers[i].v_bias_term(vs[i])
            v_weight += self.v_layers[i].v_weight_term(vs[i])

        return -(v_free + v_bias + v_weight * h + T.dot(h, self.b_h))

    def free_energy(self, v):
        v_free = 0
        v_bias = 0
        v_weight = 0

        vs = self._vs(v)
        for i in xrange(len(vs)):
            v_free += self.v_layers[i].v_free_term(vs[i])
            v_bias += self.v_layers[i].v_bias_term(vs[i])
            v_weight += self.v_layers[i].v_weight_term(vs[i])

        h_term = T.sum(T.log(1 + T.exp(v_weight + self.b_h)), axis=1)
        return -(v_bias + v_free + h_term)

    def v_given_h(self, h):
        vs_stat = []
        for i in xrange(len(self.v_layers)):
            vs_stat.append(self.v_layers[i].v_given_h(h))
        return sparse.structured_dot(T.concatenate(vs_stat, axis=1), self.big_mask)

    def h_given_v(self, v):
        vs = self._vs(v)

        v_weight = 0
        for i in xrange(len(vs)):
            v_weight += self.v_layers[i].v_weight_term(vs[i])

        p_h_v = T.nnet.sigmoid(v_weight + self.b_h)
        return p_h_v

    # vs0_sample is  list contain samples of each v_type
    def sample_h_given_v(self, v0_sample):
        h1_mean = self.h_given_v(v0_sample)
        h1_sample = self._mrng.binomial(size=h1_mean.shape, n=1, p=h1_mean,
                                        dtype=theano.config.floatX)
        return [h1_mean, h1_sample]

    # sample vs1 given h0_sample
    def sample_v_given_h(self, h0_sample):
        vs_stat = []
        vs_sample = []

        for i in xrange(len(self.v_layers)):
            v1_stat, v1_sample = self.v_layers[i].sample_v_given_h(h0_sample)

            vs_stat.append(v1_stat)
            vs_sample.append(v1_sample)

        v_stat = sparse.structured_dot(T.concatenate(vs_stat, axis=1), self.big_mask)
        v_sample = sparse.structured_dot(T.concatenate(vs_sample, axis=1), self.big_mask)

        return [v_stat, v_sample]

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
        CD_fn = theano.function([start_h], outputs=outputs[-1], updates=updates)
        return CD_fn(data_h)

    def run_CD_from_v(self, k, data_v):
        start_v = T.matrix("start_v")
        # [h_means, h_samples, v_stats, v_samples], updates \
        outputs, updates \
            = theano.scan(fn=self.gibbs_vhv, outputs_info=[None, None, None, start_v],
                          n_steps=k, name="gibbs_vhv")
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
                          name='gibbs_vhv')
        updates.update({persis_h: h_samples[-1]})
        if is_sample:
            gibbs_fn = theano.function([], v_samples[-1], updates=updates, name=name)
        else:
            gibbs_fn = theano.function([], v_stats[-1], updates=updates, name=name)
        return gibbs_fn

    def sample_given_input(self, input_x, init_gibbs=1000, betw_gibbs=100, loops=10, is_sample=False):
        print "Sample data from input using model {}".format(self.name)
        # Neu kich thuoc input la 1 thi phai chuyen no ve kich thuoc 2
        if len(input_x.shape) == 1:
            persis_v = theano.shared(np.asarray(input_x.reshape(1, input_x.shape[0]),
                                                dtype=theano.config.floatX))
        else:
            persis_v = theano.shared(np.asarray(input_x, dtype=theano.config.floatX))

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
        print "Sample random data using model {}".format(self.name)
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

    def get_viewed_cost(self, v0, v_stat):
        cost = 0

        vs0 = self._vs(v0)
        vs_stat = self._vs(v_stat)

        for i in xrange(len(self.v_layers)):
            type_cost = self.v_layers[i].get_viewed_cost(vs0[i], vs_stat[i])
            cost += type_cost

        return cost

    def nll_grad_formula(self, v0, vk):
        h0 = self.h_given_v(v0)
        hk = self.h_given_v(vk)

        gb_h = T.mean(hk - h0, axis=0)
        grads = [gb_h]

        vs0 = self._vs(v0)
        vsk = self._vs(vk)

        for i in xrange(len(self.v_layers)):
            grads.extend(self.v_layers[i].nll_grad_formula(vs0[i], vsk[i], h0, hk))

        return grads

    def l1_grad(self, l1):
        grads = [0]
        for i in xrange(len(self.v_layers)):
            grads.extend(self.v_layers[i].l1_grad(l1))
        return grads

    def l2_grad(self, l2):
        grads = [0]
        for i in xrange(len(self.v_layers)):
            grads.extend(self.v_layers[i].l2_grad(l2))
        return grads

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
            l1_grads = self.l1_grad(l1)
            grads = [grads[i] + l1_grads[i] for i in xrange(len(self.params))]

            if store_grad:
                print "\nGradients over L1 regularization are stored in l1_grads"
                l1_shared_grads, updates = store_grads_in_update(self.params, l1_grads, updates)
                self.stored_grads['l1_grads'] = l1_shared_grads

        if l2 is not None:
            print "Add L2 regularization ({}) to parameter updates".format(l2)
            l2_grads = self.l2_grad(l2)
            grads = [grads[i] + l2_grads[i] for i in xrange(len(self.params))]

            if store_grad:
                print "\nGradients over L2 regularization are stored in l2_grads"
                l2_shared_grads, updates = store_grads_in_update(self.params, l2_grads, updates)
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
        persis_h = theano.shared(persis_h_data, borrow=True) \
            if persis_h_data is not None else None

        lr = T.scalar('lr')
        cost, updates = self.get_cost_udpates(lr, k, persis_h, l1, l2, stable_update, store_grad)
        print "\nBuild computation graph for training function of model {}".format(self.name)
        self.train_fn = theano.function([self.input, lr], cost, updates=updates)

        rv = self.v_given_h(self.h_given_v(self.input))
        test_cost = self.get_viewed_cost(self.input, rv)
        test_cost = T.mean(test_cost)
        print "Build computation graph for validation function of model {}".format(self.name)
        self.valid_fn = theano.function([self.input], test_cost)