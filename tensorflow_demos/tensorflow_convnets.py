import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def cnn_model(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs = input_layer,
                             filters = 32,
                             kernel_size = [5, 5],
                             padding = 'same',
                             activation = tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout=tf.layers.dropout(inputs=dense, rate = 0.4, training=mode==tf.estimator.ModeKeys.TRAIN)
    predictions = tf.layer.dense(inputs = dropout, units=10)