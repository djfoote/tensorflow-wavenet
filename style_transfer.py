import wavenet.model as wavenet

import tensorflow as tf
import numpy as np

try:
    reduce
except NameError:
    from functools import reduce


def style_transfer(
        network,
        content,
        style,
        iterations,
        content_weight,
        style_weight,
        learning_rate):
    """
    """
    # content features for content signal
    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        signal = tf.placeholder('float', shape=(1,) + content.shape)
        net = network._create_network(signal)
        content_features = net.outputs[CONTENT_LAYER].eval(
            feed_dict={signal: np.array([content])})

    # style features for style signal
    style_features = {}
    with g.as_default(), tf.Session() as sess:
        signal = tf.placeholder('float', shape=(1,) + style.shape)
        net = network._create_network(signal)
        for layer in STYLE_LAYERS:
            features = net.outputs[layer].eval(feed_dict={signal: np.array([style])})
            # Might need to reshape
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    # define signal variable
    with tf.Graph.as_default():
        noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
        initial = tf.random_normal(shape) * 0.256
        signal = tf.Variable(initial)
        net = network.create_network(signal)

        # content loss
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net.outputs[CONTENT_LAYER] - content_features) / content_features.size)

        # style loss
        style_loss = 0
        for layer in STYLE_LAYERS:
            x_layer = net.outputs[layer]
            # From here...
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = style_features[i][style_layer]
            # ... to here is probably totally wrong; I haven't worked out the actual shapes
            style_loss += style_weight * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size

    # overall loss
    loss = content_loss + style_loss

    # optimizer setup
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # optimization
    # TODO
