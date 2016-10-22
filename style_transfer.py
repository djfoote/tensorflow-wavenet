from wavenet import WaveNetModel, load_audio_files, mu_law_decode
from generate import write_wav

import json
import argparse

import tensorflow as tf
import numpy as np

WAVENET_PARAMS = './davis_wavenet_params.json'
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
        output_path,
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
            content_features = wavenet_model.layer_responses(signal)[CONTENT_LAYER].eval(
                feed_dict={signal: np.array([content])})

        # style features for style signal
        # style_features = {}
        # with tf.Session() as sess:
        #     print("Restoring model from checkpoint...")
        #     saver.restore(sess, checkpoint)
        #     print("Computing style features...")
        #     signal = tf.placeholder('float', shape=(1,) + style.shape)
        #     layer_responses = wavenet_model.layer_responses(signal)
        #     style_targets = list(np.array(layer_responses)[STYLE_LAYERS])
        #     start_time = time.time()
        #     all_features = sess.run(style_targets, feed_dict={signal: np.array([style])})
        #     for layer, features in zip(STYLE_LAYERS, all_features):
        #         features = features.reshape(-1, features.shape[-1])
        #         gram = np.matmul(features.T, features) / features.size
        #         style_features[layer] = gram

        print("Setting up optimization problem...")
        # define signal variable and set up backprop
        # Might need new graph? Hopefully handled by var_list for optimizer
        initial = tf.random_normal((1,) + content.shape)
        initial, _ = wavenet_model.preprocess_input(initial)
        signal = tf.Variable(initial, name='result')
        layer_responses = wavenet_model.layer_responses(signal, preprocess=False)
        # The preprocessing step is not differentiable

        # content loss
        content_loss = 2 * tf.nn.l2_loss(
            layer_responses[CONTENT_LAYER] - content_features) / content_features.size

        # style loss
        # style_loss = 0
        # for style_layer in STYLE_LAYERS:
        #     x_layer = layer_responses[style_layer]
        #     # From here...
        #     _, height, width, number = map(lambda i: i.value, x_layer.get_shape())
        #     size = height * width * number
        #     feats = tf.reshape(x_layer, (-1, number))
        #     gram = tf.matmul(tf.transpose(feats), feats) / size
        #     style_gram = style_features[style_layer]
        #     # ... to here is probably totally wrong; I haven't worked out the actual shapes
        #     style_loss += 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size

        # overall loss
        # loss = content_weight * content_loss + style_weight * style_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(content_loss, var_list=[signal])

        # optimization
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, checkpoint)
            for i in np.arange(iterations):
                train_step.run()
                print(i+1)
                print('loss: {}'.format(content_loss.eval()))
                if i % 10 == 0:
                    reshaped = tf.reshape(signal, (-1, wavenet_model.quantization_channels))
                    audio = mu_law_decode(
                        reshaped.eval().argmax(axis=1), wavenet_model.quantization_channels).eval()
                    # not_one_hot = np.where(reshaped.eval())[1]
                    # audio = mu_law_decode(
                    #     not_one_hot, wavenet_model.quantization_channels).eval()
                    print(audio)
                    write_wav(audio, SAMPLE_RATE, output_path)


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
        output_path=args.output,
        iterations=ITERATIONS,
        content_weight=CONTENT_WEIGHT,
        style_weight=STYLE_WEIGHT,
        learning_rate=LEARNING_RATE)

if __name__ == '__main__':
    main()
