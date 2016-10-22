from __future__ import print_function

from wavenet import load_audio_files

import argparse
import generate

import tensorflow as tf
import numpy as np

WAVENET_PARAMS = './mostafa_wavenet_params.json'
CONTENT_WEIGHT = 1e1
STYLE_WEIGHT = 1e-5
ITERATIONS = 10000
LEARNING_RATE = 1e-3
CONTENT_LAYER = 15
STYLE_LAYERS = [1, 5, 9, 13, 17]  # Kind of arbitrary at this point
SAMPLE_RATE = 16000
STYLE_SHIFTS = 20


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('content', type=str)
    parser.add_argument('style', type=str)
    parser.add_argument('output', type=str)
    return parser.parse_args()


def basic_transfer(content_signal, style_signal, output):
    with tf.Graph().as_default():  # Sublimelinter can't handle multiple statements in a 'with'
        with tf.Session() as sess:
            print("Computing content and style features...")
            content_features = content_signal
            style_features = get_style_features(style_signal, STYLE_SHIFTS).eval()

            print("Creating loss function...")
            signal = tf.Variable(tf.random_normal(content_signal.shape))
            signal_style_features = get_style_features(signal, STYLE_SHIFTS)

            content_loss = tf.nn.l2_loss(signal - content_features) / content_features.size
            style_loss = tf.nn.l2_loss(signal_style_features - style_features) / style_features.size
            loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

            print("Initializing optimizer...")
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, var_list=[signal])
            sess.run(tf.initialize_all_variables())

            print("Beginning training")
            for i in np.arange(ITERATIONS):
                train_step.run()
                if i % 1000 == 0:
                    print("iteration number: ", i)
                    print("  content loss:", CONTENT_WEIGHT * content_loss.eval())
                    print("    style loss:", STYLE_WEIGHT * style_loss.eval())
                    print("    total loss:", loss.eval())
                    generate.write_wav(signal.eval(), SAMPLE_RATE, output)
            generate.write_wav(signal.eval(), SAMPLE_RATE, output)


def get_style_features(signal, shifts):
    padded = tf.pad(signal, [[0, shifts]])
    reshaped = tf.reshape(padded, (1, -1, 1))
    conv_filter = tf.reshape(signal, (-1, 1, 1))
    conv = tf.nn.conv1d(reshaped, conv_filter, 1, 'VALID')
    return tf.reshape(conv, (-1,))


def main():
    args = get_arguments()
    content_signal, style_signal = load_audio_files([args.content, args.style], SAMPLE_RATE)
    content_signal = np.reshape(content_signal[:100000], (-1,))
    style_signal = np.reshape(style_signal[:100000], (-1,))
    basic_transfer(
        content_signal=content_signal,
        style_signal=style_signal,
        output=args.output)

if __name__ == '__main__':
    # main()
    args = get_arguments()
    content_signal, style_signal = load_audio_files([args.content, args.style], SAMPLE_RATE)
    content_signal = np.reshape(content_signal[:100000], (-1,))
    style_signal = np.reshape(style_signal[:100000], (-1,))
    basic_transfer(
        content_signal=content_signal,
        style_signal=style_signal,
        output=args.output)
