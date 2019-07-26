from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print (train_images.shape)

train_images = train_images/255.
test_images = test_images/255.

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(128, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)
                          ])

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.compile(optimizer = 'Adam',tf.losses.
#             loss = tf.losses.CategoricalCrossentropy(from_logits=True),
#             metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 10, steps_per_epoch=5, batch_size = 100)


test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy: ', test_acc)

predictions = model.predict(test_images)
print (np.argmax(predictions[0]))

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)

    return x, y

def create_dataset(xs, ys, n_classes = 10):
    ys = tf.one_hot(ys, depth = n_classes)

    return tf.data.Dataset.from_tensor_slices((xs, ys)).map(preprocess) \
    .shuffle(len(ys))\
    .batch(128)


