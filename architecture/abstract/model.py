import numpy as np
import theano
from collections import OrderedDict
from theano.misc.pkl_utils import load, dump
from time import time

import logging
import sys

from utils.general_utils import iterate_minibatch_indices
from lr_decay import LR_Decay


class TrainMonitor(object):
    def __init__(self, model):
        # architecture should be subclass of Model
        assert issubclass(model.__class__, Model), \
            "`architecture` argument must be subclass of architecture.abstract.architecture.Model"
        self.model = model

    def pre_epoch(self, epoch_id, train_data):
        pass

    def post_epoch(self, epoch_id, train_data):
        pass

    def pre_batch(self, epoch_id, batch_id, train_batch):
        pass

    def post_batch(self, epoch_id, batch_id, train_batch):
        pass


class Model(object):
    def __init__(self, name=''):
        # Config logger
        self.logger = logging.getLogger('model_logger')
        # Ensure that multiple instances will use just one handler
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)

        stream_hdlr = logging.StreamHandler(stream=sys.stdout)
        stream_hdlr.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(stream_hdlr)

        self.name = name
        self.params = []
        self.train_monitors = []
        self.stored_grads = OrderedDict()

        self.train_fn = None
        self.valid_fn = None

        # Training
        self.learning_algor = None
        self.learning_config = {}

    def get_save(self):
        raise NotImplementedError('get_save() method must be overridden')

    def save(self, filename):
        with open(filename, 'wb') as f:
            dump(self.get_save(), f)
        print "\nSuccessfully saved architecture to file: %s" % filename

    def set_load(self, saved_data):
        raise NotImplementedError('get_save() method must be overridden')

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            saved = load(f)
        print "\nSuccessfully loaded architecture from file: %s" % filename
        return saved

    def load(self, filename):
        saved_data = Model.load_model(filename=filename)
        return self.set_load(saved_data)

    def shared_from_params(self):
        shared = []
        for param in self.params:
            shared.append(theano.shared(np.zeros(param.shape.eval(), dtype=theano.config.floatX),
                                        borrow=True))
        return shared

    def print_params_info(self):
        print "\nGeneral information of {}'s parameters:".format(self.name)
        for param in self.params:
            value = param.get_value(borrow=True)
            print "{0} | Max: {1}, Min: {2}, NaN: {3}".format(param.name, abs(value).max(), abs(value).min(),
                                                              np.isnan(value).any())

    def print_grads_info(self):
        print "\nGeneral information of {}'s parameter gradients:".format(self.name)
        for grads_key, grads in self.stored_grads.iteritems():
            print "\n{}:".format(grads_key)
            for i, grad in enumerate(grads):
                value = grad.get_value(borrow=True)
                print "{0} | Max: {1}, Min: {2}, NaN: {3}".format(grad.name,
                                                                  abs(value).max(), abs(value).min(),
                                                                  np.isnan(value).any())

    def register_train_monitor(self, train_monitor):
        # positions is a list of four values pre_epoch, post_epoch, pre_batch, post_batch
        if self.train_monitors is None or len(self.train_monitors) == 0:
            self.train_monitors = [train_monitor]
        else:
            self.train_monitors.append(train_monitor)

    def enable_grads_view(self, batch_level=True):
        class GradsMonitorBatch(TrainMonitor):
            def post_batch(self, epoch_id, batch_id, train_batch):
                print "#" + "-" * 100 + "#"
                self.model.print_params_info()
                self.model.print_grads_info()
                print

        class GradsMonitorEpoch(TrainMonitor):
            def post_epoch(self, epoch_id, train_data):
                print "#" + "-" * 100 + "#"
                self.model.print_params_info()
                self.model.print_grads_info()
                print
        if batch_level:
            self.register_train_monitor(GradsMonitorBatch(self))
        else:
            self.register_train_monitor(GradsMonitorEpoch(self))
        #self.train_monitor = GradsMonitorBatch(self) if batch_level else GradsMonitorEpoch(self)

    def config_train(self, **kwargs):
        raise NotImplementedError('config_train() method must be overridden')

    def set_learning_algor(self, algor, **config):
        self.learning_algor = algor
        self.learning_config = config

    def check_learning_algor(self):
        if self.learning_algor is None:
            print "\nLearning algorithm is not set"
            return False
        else:
            print "\nLearning algorithm {} is set with configuration {}". \
                format(self.learning_algor, self.learning_config)
            return True

    def train(self, train_x, valid_x=None, lr=0.02, lr_decay=None,
              n_epochs=300, batch_size=100, valid_all=False, save_freq=10,
              save_file=None, log_file=None, err_file=None, auto_config=True,
              **kwargs):

        if auto_config:
            self.config_train(**kwargs)
        self.check_learning_algor()

        train_errs = []
        best_valid_err = np.inf

        train_indices = np.arange(len(train_x))  # train indices
        if self.valid_fn is not None and valid_x is not None:
            valid_indices = np.arange(len(valid_x))  # valid indices

        # File writer to save training and validation error
        if err_file is not None:
            err_f_log = open(err_file, 'w')
            err_f_log.write('epoch,train_error')

            if self.valid_fn is not None and valid_x is not None:
                err_f_log.write(',valid_error')
            err_f_log.close()

        if log_file is not None:
            file_hdlr = logging.FileHandler(filename=log_file, mode='w')
            file_hdlr.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(file_hdlr)

        # print "Start training..."
        self.logger.info("\nStart training architecture {}...".format(self.name))

        for epoch in xrange(n_epochs):
            ep_start = time()

            self.logger.info("Training epoch {}".format(epoch))

            for monitor in self.train_monitors:
                monitor.pre_epoch(epoch, train_x)

            ep_err = 0
            mb_count = 0
            for mb_indices in iterate_minibatch_indices(train_indices, batch_size, True):
                for monitor in self.train_monitors:
                    monitor.pre_batch(epoch, mb_count, train_x)

                mb_err = self.train_fn(train_x[mb_indices], lr)

                for monitor in self.train_monitors:
                    monitor.post_batch(epoch, mb_count, train_x)

                ep_err += mb_err
                mb_count += 1

            ep_err = 1.0 * ep_err / mb_count
            train_errs.append(ep_err)

            ep_end = time()

            self.logger.info(
                "Epoch {0}'s error: {1:.4f}, training time: {2:.4f}s, lr: {3:.6f}"
                    .format(epoch, ep_err, ep_end - ep_start, lr))
            if err_file is not None:
                err_f_log = open(err_file, 'a')
                err_f_log.write('\n{0},{1:.4f}'.format(epoch, ep_err))
                err_f_log.close()

            for monitor in self.train_monitors:
                monitor.post_epoch(epoch, train_x)

            # Learning rate change here
            if lr_decay is not None and issubclass(lr_decay.__class__, LR_Decay):
                lr = lr_decay.decay(lr, epoch)

            # Validation
            if self.valid_fn is not None and valid_x is not None:
                if valid_all:
                    valid_err = self.valid_fn(valid_x)
                    valid_err = valid_err.sum()
                else:
                    valid_err = 0
                    mb_count = 0
                    for mb_indices in iterate_minibatch_indices(valid_indices, batch_size, True):
                        valid_err += self.valid_fn(valid_x[mb_indices])
                        mb_count += 1

                    valid_err = 1.0 * valid_err / mb_count

                self.logger.info("Valid error: {0:.4f} | Best valid error: {1:.4f}"
                                 .format(valid_err, best_valid_err))

                if err_file is not None:
                    err_f_log = open(err_file, 'a')
                    err_f_log.write(',{0:.4f}'.format(valid_err))
                    err_f_log.close()

                if valid_err < best_valid_err:
                    best_valid_err = valid_err

            if save_file is not None and epoch % save_freq == 0:
                self.save(save_file)

            self.logger.info('-' * 50)

        self.logger.info("Finish training architecture {}".format(self.name))
        if log_file is not None:
            self.logger.removeHandler(file_hdlr)

        return train_errs
