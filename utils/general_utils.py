import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from PIL import Image

import theano


def error_plot(err_file, saved_img_file=None):
    err_df = pd.read_csv(err_file, sep=',')
    err_df.plot(kind='line', x='epoch', legend=True)
    plt.gcf().set_size_inches(15, 10)

    print "\nFinish plotting training/validation curve"
    if saved_img_file is not None:
        plt.savefig(saved_img_file, dpi=100)
    else:
        plt.show()


def plot_3D_embedding(data, labels, cmap, title=None, save_file=None):
    classes = np.unique(labels)
    color_map = {classes[i]: cmap(1.0 * i / len(classes)) for i in xrange(len(classes))}

    classes_ids = []
    for i in xrange(len(classes)):
        classes_ids.append(np.arange(len(labels))[labels == classes[i]])

    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')

    colors = [color_map.get(labels[i]) for i in xrange(len(labels))]
    ax3D.scatter(data[:, 0], data[:, 1], data[:, 2],
                 c=colors, marker='o', edgecolors='black')

    legend_patches = [mpatches.Patch(color=value, label=key) \
                      for key, value in color_map.iteritems()]
    plt.legend(handles=legend_patches, loc='upper right', ncol=2)

    fig.set_size_inches(15, 10)

    if title is not None:
        ax3D.set_title(title)
    if save_file is not None:
        plt.savefig(save_file, dpi=1000)
    else:
        plt.show()


def plot_2D_embedding(data, labels, cmap, title=None, save_file=None):
    classes = np.unique(labels)
    color_map = {classes[i]: cmap(1.0 * i / len(classes)) for i in xrange(len(classes))}

    classes_ids = []
    for i in xrange(len(classes)):
        classes_ids.append(np.arange(len(labels))[labels == classes[i]])

    fig, ax = plt.subplots()
    for i in xrange(len(classes)):
        ax.scatter(data[classes_ids[i], 0], data[classes_ids[i], 1],
                   color=color_map.get(classes[i]), marker='o', edgecolors='black', label='{}'.format(classes[i]))

    ax.legend(prop={'size': 24})
    fig.set_size_inches(15, 10)

    if title is not None:
        ax.set_title(title)
    if save_file is not None:
        plt.savefig(save_file, dpi=1000)
    else:
        plt.show()


def scale_to_unit_interval(vals, eps=1e-8):
    vals = vals.copy()
    vals -= vals.min()
    vals *= 1.0 / (vals.max() + eps)
    return vals


def init_weight(in_dim, out_dim, value=None, name=''):
    if value is not None:
        if len(value.shape) == 2 and value.shape[0] == in_dim and value.shape[1] == out_dim:
            init_value = np.asarray(value, dtype=theano.config.floatX)
        else:
            raise ValueError("The shape of weight " + name + " is invalid.")
    else:
        init_value = np.asarray(np.random.uniform(
                                        low=-4.0 * np.sqrt(6.0 / (out_dim + in_dim)),
                                        high=4.0 * np.sqrt(6.0 / (out_dim + in_dim)),
                                        size=(in_dim, out_dim)
                                        ),
                                dtype=theano.config.floatX
                                )
    return theano.shared(value=init_value, name=name, borrow=True)


def init_bias(dim, value=None, name=''):
    if value is not None:
        if len(value.shape) == 1 and value.shape[0] != dim:
            init_value = np.asarray(value, dtype=theano.config.floatX)
        else:
            raise ValueError("The shape of bias " + name + " is invalid.")
    else:
        init_value = np.zeros(dim, dtype=theano.config.floatX)
    return theano.shared(value=init_value, name=name, borrow=True)


# weight_shape is (n_out_channels, n_in_channels, w_width, w_height)
def init_conv_weight(weight_shape, pool_size=(1, 1), value=None, name=''):
    if value is not None:
        assert np.array_equal(weight_shape, value.shape)
        init_value = np.asarray(value, dtype=theano.config.floatX)
    else:
        fan_in = np.prod(weight_shape[1:])
        fan_out = (weight_shape[0] * weight_shape[2] * weight_shape[3]) // np.prod(pool_size)

        W_bound = np.sqrt(6.0 / (fan_in + fan_out))
        init_value = np.asarray(np.random.uniform(
                            low=-W_bound, high=W_bound, size=weight_shape),
                            dtype=theano.config.floatX)
    return theano.shared(value=init_value, name=name, borrow=True)


# Bias shape is
def init_conv_bias(bias_shape, value=None, name=''):
    if value is not None:
        init_value = np.asarray(value, dtype=theano.config.floatX)
    else:
        init_value = np.zeros(bias_shape, dtype=theano.config.floatX)

    return theano.shared(value=init_value, name=name, borrow=True)


def shared_dataset(data, dtype=theano.config.floatX, borrow=True):
    shared_data = theano.shared(np.asarray(data,
                                        dtype=theano.config.floatX),
                                        borrow=borrow)

    if dtype==theano.config.floatX:
        return shared_data
    else:
        return theano.tensor.cast(shared_data, dtype)


def iterate_minibatch_indices(indices, batch_size, shuffle=True):
    if shuffle:
        np.random.shuffle(indices)

    n_batches = len(indices) / batch_size
    for mb_index in xrange(n_batches):
        yield indices[mb_index * batch_size: (mb_index + 1) * batch_size]


def iterate_minibatches(inputs, batch_size, shuffle=True):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt]


def draw_image(data, img_shape=(28, 28), tile_shape=(10, 10),
               tile_spacing=(1, 1), save_file=None,
               background_intensity=0.5, mode=None):
    print "If img_shape is a 3-tuple, the first element must be image channels. " \
          "The next two elements are image height and image width."
    data = np.asarray(data)
    data = np.reshape(data, (data.shape[0], np.prod(data.shape[1:])))

    # Chu y la Theano de channel dang truoc
    if len(img_shape) == 3:
        assert img_shape[0] == 1 or img_shape[0] == 3, \
            "The number of image channels must be 1 or 3. Found {}".format(img_shape[0])
    elif len(img_shape) == 2:
        img_shape = (1, ) + img_shape

    img_channel, img_h, img_w = img_shape
    tile_h, tile_w = tile_shape
    space_h, space_w = tile_spacing

    # Do 2 dau khong can space
    out_shape = ((img_h + space_h) * tile_h - space_h,
                 (img_w + space_w) * tile_w - space_w, img_channel)

    out_array = background_intensity * np.ones(out_shape, dtype=np.uint8)
    # Loop over with then h
    for ty in xrange(tile_h):
        for tx in xrange(tile_w):
            # If we can draw the weight
            if ty * tile_w + tx < data.shape[0]:
                img_data = data[ty * tile_w + tx].reshape(img_shape)

                # Channel is the first dim in Theano while it is the last in PIL
                img_data = np.transpose(img_data, (1, 2, 0))
                img_data = scale_to_unit_interval(img_data)

                out_array[ty * (img_h + space_h): ty * (img_h + space_h) + img_h,
                tx * (img_w + space_w): tx * (img_w + space_w) + img_w, :] = img_data

    out_array = (out_array * 255).astype(np.uint8)
    if mode is None:
        if img_channel == 1:
            mode = 'L'
            out_array = out_array[:, :, 0]
        else:
            mode = 'RGB'
    image = Image.fromarray(out_array, mode)

    if save_file is not None:
        image.save(save_file)
    return image


def draw_bias(b, img_shape=(28, 28), save_file=None):
    img_array = b.reshape(img_shape)
    img_array = scale_to_unit_interval(img_array)

    image = Image.fromarray(img_array)
    if save_file is not None:
        image.save(save_file)
    return image


def draw_weight(W, img_shape=(28, 28), tile_shape=(10, 10),
                tile_spacing=(0, 0), save_file=None,
                background_intensity=0.5, mode=None):

    img_h, img_w = img_shape
    assert img_h * img_w == W.shape[0], "The total image size (height * width) is not equal to " \
        "W.shape[0] ({} and {} respectively)".format(img_h * img_w, W.shape[0])

    tile_h, tile_w = tile_shape
    assert tile_h * tile_w <= W.shape[1], "The total tile size (height * width) must be smaller than " \
        "or equal to W.shape[1] ({} and {} respectively)".format(tile_h * tile_w, W.shape[1])

    space_h, space_w = tile_spacing

    # Do 2 dau khong can space
    out_shape = [(img_h + space_h)* tile_h - space_h,
                 (img_w + space_w)* tile_w - space_w]

    out_array = background_intensity * np.ones(out_shape, dtype=np.uint8)
    # Loop over with then h
    for ty in xrange(tile_h):
        for tx in xrange(tile_w):
            weight = W[:, ty * tile_w + tx].reshape(img_shape)
            weight = scale_to_unit_interval(weight)

            out_array[ty*(img_h + space_h): ty*(img_h + space_h) + img_h,
                      tx*(img_w + space_w): tx*(img_w + space_w) + img_w] = weight

    out_array = (out_array * 255).astype(np.uint8)
    if mode is None:
        if len(img_shape) == 2 or (len(img_shape) == 3 and img_shape[0] == 1):
            mode = 'L'
        else:
            mode = 'RGB'
    image = Image.fromarray(out_array, mode)

    if save_file is not None:
        image.save(save_file)
    return image