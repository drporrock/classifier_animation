import numpy as np
from sklearn.datasets.samples_generator import make_moons, make_circles

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_noisy_data(samples, noise=0.05):
    # xy, labels = make_moons(n_samples = samples, noise=noise)
    xy, labels = make_circles(n_samples = samples, noise=noise)
    # xy = xy / xy.max()
    return xy, labels


def _initialise_params(neurons, inputs):
    """
    Return initialised w and b

    Input is dictionary containing the neuron_dims, an array
    :param neurons: Number of neurons in layer
    :param inputs: Size of input layer
    :return: Arrays containing W and b respectively
    """
    w = np.random.randn(neurons, inputs) * np.sqrt(2.0/inputs)
    b = np.zeros((neurons, 1))
    assert w.shape == (neurons, inputs)
    return w, b


def initialise_params_deep(layer_dims, input, m):
    """
    Initialise parameters for deep network

    :param layer_dims: List of the number of neurons per layer
    :param input_size: Size of input
    :return: Dict containing initialised parameters
    """
    params = {}
    layer_labels = range(1, len(layer_dims) + 1)
    a_prev_shape = input.shape
    prev_layer = input.shape[1]
    for l in layer_labels:
        w, b = _initialise_params(layer_dims[l-1], prev_layer)
        params[f'W{l}'] = w
        params[f'b{l}'] = b
        # a_prev = layer_dims[l-1]
        # a_prev = m
        prev_layer = layer_dims[l-1]
        # a_prev_shape = (layer_dims[l-1], a_prev_shape[0])
    return params


def _relu(z):
    return np.maximum(0, z)


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _linear_forward(a_prev, w, b, activation):
    assert activation in ('relu', 'sigmoid'), 'Activation should be relu or sigmoid'
    z = np.dot(w, a_prev) + b
    a = _relu(z) if activation == 'relu' else _sigmoid(z)
    # assert z.shape == (w.shape[0], a_prev.shape[0])
    # assert a.shape == z.shape
    return a, z


def _cost(a, y):
    m = len(y)
    cost = -np.sum(np.dot(y.T, np.log(a.T)) + np.dot((1 - y.T), np.log(1 - a.T))) / m
    return cost


def _initialise_back_prop(al, y):
    da = -(np.divide(y, al) - np.divide(1 - y, 1 - al))
    # da = -(y / al - (1 - y) / (1 - al))
    return da


def _linear_activation_back(da, z, activation):
    assert activation in ('relu', 'sigmoid'), 'Activation should be relu or sigmoid'
    if activation == 'relu':
        dz = da.copy()
        # dz = z.copy()
        dz[z <= 0] = 0
        # dz[z > 0] =
    else:
        s = 1 / (1 + np.exp(-z))
        dz = da * s * (1 - s)
    return dz


def _linear_backward(dz, a_prev, w):
    m = len(a_prev)
    dw = np.dot(dz, a_prev.T) / m
    db = np.sum(dz, axis=1, keepdims=True) / m
    da_prev = np.dot(w.T, dz)
    return dw, db, da_prev


def update_params(params, grads, learning_rate):
    layers = len(grads) // 2
    for l in range(1, layers + 1):
        params[f'W{l}'] = params[f'W{l}'] - grads[f'dW_{l}'] * learning_rate
        params[f'b{l}'] = params[f'b{l}'] - grads[f'db_{l}'] * learning_rate
    return params


def forward_loop(x, neurons, params):
    a = x.T
    layers = len(neurons)
    cache = {}
    for l in range(1, layers):
        w = params[f'W{l}']
        b = params[f'b{l}']
        a_prev = a
        a, z = _linear_forward(a_prev, w, b, 'relu')
        cache[f'a{l}'] = a
        cache[f'z{l}'] = z
    al, zl = _linear_forward(a, params[f'W{layers}'], params[f'b{layers}'], 'sigmoid')
    cache[f'a{layers}'] = al
    cache[f'z{layers}'] = zl
    return al, zl, cache


def backward_loop(al, zl, y, cache, params, neurons):
    grads = {}
    # layers = len(cache) // 2 + 1
    layers = len(neurons)
    dal = _initialise_back_prop(al, y.T)
    dz = _linear_activation_back(dal, zl, 'sigmoid')
    w = params[f'W{layers}']
    a_prev = cache[f'a{layers - 1}']
    dw, db, da_prev = _linear_backward(dz, a_prev, w)
    grads[f'dW_{layers}'] = dw
    grads[f'db_{layers}'] = db
    for l in reversed(range(1, layers)):
        z = cache[f'z{l}']
        a_prev = cache[f'a{l-1}']
        w = params[f'W{l}']
        dz = _linear_activation_back(da_prev, z, 'relu')
        dw, db, da_prev = _linear_backward(dz, a_prev, w)
        grads[f'dW_{l}'] = dw  # Should this be l-1?
        grads[f'db_{l}'] = db
    return grads


def main():
    xy, labels = generate_noisy_data(50)
    labels = labels.reshape((len(labels), 1))
    m = xy.shape[0]
    input_dims = xy.shape[1]
    neurons = [100, 50, 5, 1]
    params = initialise_params_deep(neurons, xy, m)
    costs = []
    for i in range(0, 1000):
        al, zl, cache = forward_loop(xy, neurons, params)
        cache['a0'] = xy.T
        cost = _cost(al, labels)
        assert cost is not np.nan
        costs.append(cost)
        grads = backward_loop(al, zl, labels, cache, params, neurons)
        params = update_params(params, grads, 0.02)
        if i % 10 == 0:
            print(f'Iteration {i} cost: {cost}')
            p, al = predict(xy, labels, params, neurons)


def predict(X, y, parameters, neurons):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[0]
    # n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    # probas, caches = L_model_forward(X, parameters)
    al, zl, cache = forward_loop(X, neurons, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, al.shape[1]):
        if al[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))
    c = ['r' if i == 1 else 'g' for i in p[0]]
    plt.scatter(X[:, 0], X[:, 1], color=c)
    plt.show()
    return p, al

if __name__ == '__main__':
    main()
