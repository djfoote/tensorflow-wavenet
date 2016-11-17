from __future__ import print_function

from wavenet import load_audio_files

import argparse
import generate

import tensorflow as tf
import numpy as np
import scipy
import librosa

WAVENET_PARAMS = './mostafa_wavenet_params.json'
CONTENT_WEIGHT = 1e1
STYLE_WEIGHT = 1e1
ITERATIONS = 50000
LEARNING_RATE = 1e-3
CONTENT_LAYER = 15
SAMPLE_RATE = 16000
STYLE_SHIFTS = 0
SILENCE_THRESHOLD = 0.5


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('content', type=str)
    parser.add_argument('style', type=str)
    parser.add_argument('output', type=str)
    return parser.parse_args()


def basic_transfer(content_signal, style_signal, output, use_frequency_domain=True):
    if use_frequency_domain:
        content_stft = librosa.core.stft(content_signal)
        content_signal = np.array([np.abs(content_stft), np.angle(content_stft)])
        style_stft = librosa.core.stft(style_signal)
        style_signal = np.array([np.abs(style_stft), np.angle(style_stft)])
        style_featurizer = get_style_frequency_features
    else:
        style_featurizer = lambda signal: get_style_features(signal, STYLE_SHIFTS)
    with tf.Graph().as_default():  # Sublimelinter can't handle multiple statements in a 'with'
        with tf.Session() as sess:
            print("Computing content and style features...")
            content_features = content_signal
            style_features = style_featurizer(style_signal).eval()

            print("Creating loss function...")
            signal = tf.Variable(tf.random_normal(content_signal.shape))
            signal = tf.Variable(content_signal)
            signal_style_features = style_featurizer(signal)

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
                    # import pdb; pdb.set_trace()
                    write_wav(signal.eval(), SAMPLE_RATE, output, use_frequency_domain)
            write_wav(signal.eval(), SAMPLE_RATE, output, use_frequency_domain)


def get_style_features(signal, max_shift):
    """
    Return pairwise correlations for all n-step pairs of samples where n ranges from 0 (magnitude)
        to max_shift.
    signal is a raw audio signal vector.
    """
    padded = tf.pad(signal, [[0, max_shift]])
    reshaped = tf.reshape(padded, (1, -1, 1))
    conv_filter = tf.reshape(signal, (-1, 1, 1))
    conv = tf.nn.conv1d(reshaped, conv_filter, 1, 'VALID')
    return tf.reshape(conv, (-1,))


def get_style_frequency_features(signal):
    """
    Return pairwise correlations between frequency bins.
    signal is an array of shape (2, num_frequency_bins, num_time_windows) where the matrix signal[0]
        is the magnitude of the STFT (i.e. the spectrogram) and signal[1] is the phase of the STFT.
    """
    # import pdb; pdb.set_trace()
    spectrogram = signal[0, :, :]
    return tf.matmul(spectrogram, spectrogram, transpose_b=True)


def write_wav(signal, sample_rate, output_file, use_frequency_domain):
    if use_frequency_domain:
        signal_stft = signal[0] * np.exp(signal[1] * 1j)
        signal = librosa.core.istft(signal_stft)
    generate.write_wav(signal, SAMPLE_RATE, output_file)


def main():
    args = get_arguments()
    import pdb; pdb.set_trace()
    content_signal, style_signal = load_audio_files([args.content, args.style], SAMPLE_RATE)
    basic_transfer(
        content_signal=content_signal.flatten(),
        style_signal=style_signal.flatten(),
        output=args.output)

if __name__ == '__main__':
    main()
    # args = get_arguments()
    # content_signal, style_signal = load_audio_files([args.content, args.style], SAMPLE_RATE)

    # Create test signal and STFT.
    # X = librosa.core.stft(x)
    # xhat = librosa.core.istft(X)
    # generate.write_wav(xhat, SAMPLE_RATE, args.output)
