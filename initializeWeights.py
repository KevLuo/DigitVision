import numpy as np

def randInitialize(layer_in, layer_out):
    w = np.zeros((layer_in, layer_out + 1))
    epsilon_init = 0.12
    w = np.random.random((layer_in, layer_out + 1)) * 2 * epsilon_init - epsilon_init
    return w