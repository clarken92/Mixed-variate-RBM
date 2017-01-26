from enum import Enum


class InputType(Enum):
    binary = 0
    gaussian = 1
    poisson = 2
    categorical = 3
    replicated_softmax = 4


class BroadInputType(Enum):
    nominal = 1
    numeric = 2