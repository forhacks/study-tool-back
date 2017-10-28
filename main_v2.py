import numpy as np
import math, random, itertools
from skimage.util.shape import view_as_blocks

# Hyperparams
layer_sizes = [1, 32, 5]
filter_size = 5
epochs = 100
batch_size = 32
learning_rate = 0.0000000005


def sigmoid(f_a):
    return 1 / (1 + np.exp(-f_a))


def im2col(img, size, stepsize=1):
    a = img
    # Parameters
    m, n = a.shape
    s0, s1 = a.strides
    nrows = m-size[0]+1
    ncols = n-size[1]+1
    shp = size[0], size[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(a, shape=shp, strides=strd)
    return out_view.reshape(size[0]*size[1], -1)[:, ::stepsize]


# Convolve layer: tested
def convolve_layer(layer, filters, new_layer_size, size=[filter_size, filter_size]):
    next_layer_size = (new_layer_size, len(layer[0]), len(layer[0][0]))
    layer = np.array([im2col(np.pad(img, ((int((size[0] - 1) / 2), int((size[0] - 1) / 2)), (int((size[0] - 1) / 2), int((size[1] - 1) / 2))), mode="constant"), size) for img in layer])
    # print(layer.shape)
    # print(filters.shape)
    # print(np.dot(filters, layer).shape)
    output_layer = np.dot(filters, layer)
    output_layer = np.sum(output_layer, axis=(0, 2)).reshape(next_layer_size)
    return output_layer


# Max pooling: tested
def max_pool(img, size):
    new_img = view_as_blocks(
        np.pad(img, ((0, (size - (len(img) % size)) % size), (0, (size - (len(img[0]) % size)) % size)),
               mode="constant"), (size, size))
    new_img = np.amax(new_img, axis=(2, 3))
    return new_img.reshape((int(math.ceil(len(img) / size)), int(math.ceil(len(img[0]) / size))))


# Max pooling layer: tested
def pool_layer(layer, size):
    return np.array([max_pool(img, size) for img in layer])


# ReLU layer: tested
def relu_layer(layer):
    gradients = np.maximum(layer, 0.01*layer)
    return gradients


# Filter initialization: tested
def init_filters():
    filters = [None]*(len(layer_sizes) - 1)
    for i in range(len(layer_sizes)-1):
        filters[i] = 0.01*np.random.randn(layer_sizes[i], layer_sizes[i + 1], filter_size * filter_size) - 0.005
    return filters


# Back-propagation for network, note: layer is derivs
def backprop_conv(prev_layer, layer):
    orig_shape = prev_layer.shape
    size = [len(layer[0]), len(layer[0][0])]
    layer = [img.flatten() for img in layer]
    prev_layer = np.array([im2col(np.pad(img, int((filter_size - 1) / 2), mode="constant"), size) for img in prev_layer])
    dw = np.zeros((len(prev_layer), len(layer), filter_size ** 2))
    for x in range(len(prev_layer)):
        for y in range(len(layer)):
            dw[x][y] = np.dot(layer[y], prev_layer[x])
    dwSum = np.rot90(np.sum(dw, axis=1), 2, axes=(0, 1))
    da = np.reshape(np.sum(np.dot(prev_layer, dwSum.T), axis=2), orig_shape)
    return [dw, da]


def backprop_relu(prev_layer, layer):
    orig_shape = prev_layer.shape
    prev_layer = [0.01 if x < 0 else 1 for img in prev_layer for row in img for x in row]
    return np.multiply(np.reshape(prev_layer, orig_shape), layer)


def backprop_pool(prev_layer, layer, deriv, size):
    repeated = np.repeat(np.repeat(layer, size, axis=2), size, axis=1)
    repeated = repeated[::, :len(prev_layer[0]):, :len(prev_layer[0][0]):]
    prev_layer = prev_layer - repeated
    prev_layer[prev_layer < 0] = 0
    prev_layer[prev_layer == 0] = 1
    repeated = np.repeat(np.repeat(deriv, size, axis=2), size, axis=1)
    repeated = repeated[::, :len(prev_layer[0]):, :len(prev_layer[0][0]):]
    prev_layer = np.multiply(prev_layer, repeated)
    return prev_layer

x = np.load('data/x.npy')
y = np.load('data/y.npy')

c = list(zip(x, y))
random.shuffle(c)

x, y = zip(*c)


filters = init_filters()

n = 1050 * layer_sizes[2]
w = 0.01 * np.random.randn(1, n)
b = 0

index = 0
for i in range(epochs):
    section = i % len(x)/batch_size
    j = 0
    for definitions in x[i * batch_size:(i + 1) * batch_size]:
        c1 = convolve_layer(np.array([definitions[0]]), filters[0], layer_sizes[1])
        p1 = pool_layer(c1, 2)
        r1 = relu_layer(p1)

        c2 = convolve_layer(r1, filters[1], layer_sizes[2])
        p2 = pool_layer(c2, 2)
        final = relu_layer(p2).flatten()

        bc1 = convolve_layer(np.array([definitions[1]]), filters[0], layer_sizes[1])
        bp1 = pool_layer(c1, 2)
        br1 = relu_layer(p1)

        bc2 = convolve_layer(r1, filters[1], layer_sizes[2])
        bp2 = pool_layer(c2, 2)
        final = np.append(final, relu_layer(p2).flatten())

        z = w.dot(final)
        a = sigmoid(z)

        j += -np.sum(y[index] * np.log(a[0]) + (1 - y[index])*np.log(1 - a[0])) / batch_size

        dz = a - y[index]
        w -= learning_rate * (dz * final)
        b -= learning_rate * np.sum(dz)

        dfinal = (w * dz).flatten()
        dfinala = np.reshape(dfinal[:n//2], (5, 7, 75))
        dr2 = backprop_relu(p2, dfinala)
        dp2 = backprop_pool(c2, p2, dr2, 2)
        dw2, dc2 = backprop_conv(r1, dp2)
        dr1 = backprop_relu(p1, dc2)
        dp1 = backprop_pool(c1, p1, dr1, 2)
        dw1, dc1 = backprop_conv(np.array([definitions[0]]), dp1)

        dfinalb = np.reshape(dfinal[n//2:], (5, 7, 75))
        dr2 = backprop_relu(bp2, dfinalb)
        dp2 = backprop_pool(bc2, bp2, dr2, 2)
        bdw2, dc2 = backprop_conv(br1, dp2)
        dr1 = backprop_relu(bp1, dc2)
        dp1 = backprop_pool(bc1, bp1, dr1, 2)
        bdw1, dc1 = backprop_conv(np.array([definitions[1]]), dp1)

        filters[1] += learning_rate * (dw2 + bdw2) / 2
        filters[0] += learning_rate * (dw1 + bdw1) / 2

        index += 1
        index %= len(y)
    print("Epoch " + str(i+1) + "/" + str(epochs) + ": Error " + str(j))
