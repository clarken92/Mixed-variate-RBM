class LR_Decay(object):
    def __init__(self, type):
        self.type = type

    def decay(self, lr, epoch):
        raise NotImplementedError("decay() method must be overridden")


class LR_StepDecay(LR_Decay):
    def __init__(self, decay_start=1, decay_step=5, decay_rate=0.2, min_lr=0.0):
        super(LR_StepDecay, self).__init__(type='step')
        self.decay_start = decay_start
        self.decay_step = decay_step

        assert decay_rate <= 1.0
        self.decay_rate = decay_rate

        self.min_lr = min_lr

    def decay(self, lr, epoch):
        if epoch < self.decay_start:
            return lr

        if epoch % self.decay_step == 0:
            return max((1.0 - self.decay_rate) * lr, self.min_lr)
        else:
            return max(lr, self.min_lr)


class LR_LinearDecay(LR_Decay):
    def __init__(self, lr0, decay_start=1, decay_rate=0.8, min_lr=0.0):
        super(LR_LinearDecay, self).__init__(type='linear')
        self.lr0 = lr0
        self.decay_start = decay_start

        assert decay_rate <= 1.0
        self.decay_rate = decay_rate

        self.min_lr = min_lr

    def decay(self, lr, epoch):
        if epoch < self.decay_start:
            return lr

        return max(self.lr0 / (1 + self.decay_rate * (epoch - self.decay_start)), self.min_lr)