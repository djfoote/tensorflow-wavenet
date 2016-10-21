from wavenet import WaveNetModel, load_audio_files

import json
import argparse

import tensorflow as tf
import numpy as np

WAVENET_PARAMS = './mostafa_wavenet_params.json'
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
ITERATIONS = 1000
LEARNING_RATE = 1e-3
CONTENT_LAYER = 15
STYLE_LAYERS = [1, 5, 9, 13, 17]  # Kind of arbitrary at this point
SAMPLE_RATE = 16000


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('content', type=str)
    parser.add_argument('style', type=str)
    parser.add_argument('output', type=str)
    return parser.parse_args()


def build_model_and_saver(wavenet_params, checkpoint):
    wavenet_model = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'])

    variables_to_restore = dict([  # Sublimelinter can't handle dict comprehensions
        (var.name[:-2], var) for var in tf.all_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)])
    saver = tf.train.Saver(variables_to_restore)
    return wavenet_model, saver


def style_transfer(
        wavenet_params,
        checkpoint,
        content,
        style,
        iterations,
        content_weight,
        style_weight,
        learning_rate):
    """
    """
    # import pdb; pdb.set_trace()
    with tf.Graph().as_default():
        print("Building model...")
        wavenet_model, saver = build_model_and_saver(wavenet_params, checkpoint)

        # content features for content signal
        with tf.Session() as sess:
            print("Restoring model from checkpoint...")
            saver.restore(sess, checkpoint)
            print("Computing content features...")
            signal = tf.placeholder('float', shape=(1,) + content.shape)
            layer_responses = wavenet_model._create_network(signal, target='layer_responses')
            import pdb; pdb.set_trace()
            content_features = layer_responses[CONTENT_LAYER].eval(
                feed_dict={signal: np.array([content])})

        # style features for style signal
        style_features = {}
        with tf.Session() as sess:
            print("Restoring model from checkpoint...")
            saver.restore(sess, checkpoint)
            print("Computing content features...")
            signal = tf.placeholder('float', shape=(1,) + style.shape)
            layer_responses = wavenet_model._create_network(signal, target='layer_responses')
            for layer in STYLE_LAYERS:
                features = layer_responses[layer].eval(feed_dict={signal: np.array([style])})
                # Might need to reshape
                gram = np.matmul(features.T, features) / features.size
                style_features[layer] = gram

        # define signal variable and set up backprop
        # Might need new graph? Hopefully handled by var_list for optimizer
        initial = tf.random_normal(content.shape) * 0.256
        signal = tf.Variable(initial)
        layer_responses = wavenet_model._create_network(signal, target='layer_responses')

        # content loss
        content_loss = 2 * tf.nn.l2_loss(
            layer_responses[CONTENT_LAYER] - content_features) / content_features.size

        # style loss
        style_loss = 0
        for style_layer in STYLE_LAYERS:
            x_layer = layer_responses[style_layer]
            # From here...
            _, height, width, number = map(lambda i: i.value, x_layer.get_shape())
            size = height * width * number
            feats = tf.reshape(x_layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = style_features[style_layer]
            # ... to here is probably totally wrong; I haven't worked out the actual shapes
            style_loss += 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size

        # overall loss
        loss = content_weight * content_loss + style_weight * style_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=[signal])

        # optimization
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            tf.initialize_variables(var_list=[signal])
            for i in np.arange(iterations):
                # TODO
                pass


def main():
    args = get_arguments()
    with open(WAVENET_PARAMS, 'r') as config_file:
        wavenet_params = json.load(config_file)

    content_signal, style_signal = load_audio_files(
        [args.content, args.style], SAMPLE_RATE)

    style_transfer(
        wavenet_params=wavenet_params,
        checkpoint=args.checkpoint,
        content=content_signal,
        style=style_signal,
        iterations=ITERATIONS,
        content_weight=CONTENT_WEIGHT,
        style_weight=STYLE_WEIGHT,
        learning_rate=LEARNING_RATE)

if __name__ == '__main__':
    main()
