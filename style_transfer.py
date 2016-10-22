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
        content = content
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
        all_signals = []
        i = 0
        initial = tf.random_normal((1,) + content.shape)
        print(initial.get_shape())
        initial, _ = wavenet_model.preprocess_input(initial)
        # content = tf.reshape(content, (-1, 1))
        with tf.Session() as sess:
            print(initial.get_shape())
            signal = tf.Variable(initial)
            tf.initialize_variables([signal]).run()
            np_signal = signal.eval()[0]
            print(np_signal.shape)
        for timestep in np_signal:
            sub_signal = tf.Variable(timestep)
            all_signals.append(sub_signal)
        signal = tf.concat(0, all_signals)
        signal = tf.reshape(signal, (1, np_signal.shape[0], np_signal.shape[1]))

        # for timestep in content:
        #     initial = tf.random_normal((1,) + timestep.shape)
        #     initial, _ = wavenet_model.preprocess_input(initial)
        #     initial = tf.reshape(initial, (-1, 1))
        #     sub_signal = tf.Variable(initial)
        #     all_signals.append(sub_signal)
        #     if i % 1000 == 0:
        #         print(i)
        #     i+=1

        # signal = tf.reshape(tf.concat(0, all_signals), (1,  wavenet_model.quantization_channels, len(content)))
        layer_responses = wavenet_model.layer_responses(signal, preprocess=False)

        content_loss = 2 * tf.nn.l2_loss(layer_responses[CONTENT_LAYER] - content_features) 

        training_steps = []
        optimizer = tf.train.AdamOptimizer(learning_rate)

        for i in np.arange(0, len(all_signals), 100):
            curr_step = optimizer.minimize(content_loss, var_list = all_signals[i:i+100])
            training_steps.append(curr_step)
            print(i, len(all_signals))
        # for sub_signal in all_signals:
        #     curr_step = optimizer.minimize(content_loss, var_list=[sub_signal])
        #     training_steps.append(curr_step)
        #     if i % 1000 == 0:
        #         print (i)

        # layer_responses = wavenet_model.layer_responses(signal, preprocess=False)
        # The preprocessing step is not differentiable

        # content loss
        # content_loss = 2 * tf.nn.l2_loss(
            # layer_responses[CONTENT_LAYER] - content_features) / content_features.size + 0.00001*tf.reduce_sum(tf.abs(signal))

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
        # train_step = tf.train.AdamOptimizer(learning_rate).minimize(content_loss, var_list=[signal])

        # optimization
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, checkpoint)
            for i in np.arange(len(training_steps)):
                for k in np.arange(500):
                    train_steps[i].run()
                    print('loss: {}'.format(content_loss.eval()))
                if i % 100 == 0:
                    reshaped = tf.reshape(signal, (-1, wavenet_model.quantization_channels))
                    audio = mu_law_decode(
                        reshaped.eval().argmax(axis=1), wavenet_model.quantization_channels).eval()
                    # not_one_hot = np.where(reshaped.eval())[1]
                    # audio = mu_law_decode(
                    #     not_one_hot, wavenet_model.quantization_channels).eval()
                    print(audio)
                    write_wav(audio, SAMPLE_RATE, output_path)
                curr_signal = all_signals[i*100:i*100+100].eval()
                max_idx = np.argmax(curr_signal, axis = 0)
                new_signal = np.zeros_like(curr_signal)
                for j in range(len(new_signal)):
                    new_signal[j, max_idx[j]] = 1
                # new_signal[max_idx] = 1
                op = all_signals[i*100:i*100+100].assign(new_signal)
                sess.run([op])



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
