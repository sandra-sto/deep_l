from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)
train_images = train_images/255.

test_images = test_images/255.

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
                          keras.layers.Dense(128, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)
                          ])

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, batch_size=100)


test_loss, test_acc = model.evaluate(test_images, test_labels)
# model.evaluate(data, labels, batch_size=32)
print('Test Accuracy: ', test_acc)

predictions = model.predict(test_images)
prediction = np.argmax(predictions[0])
print (prediction)

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.
    y = tf.cast(y, tf.int64)

    return x, y

def create_dataset(xs, ys, n_classes=10):
    ys = tf.one_hot(ys, depth=n_classes)
    dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()

    return tf.data.Dataset.from_tensor_slices((xs, ys)).map(preprocess) \
        .shuffle(len(ys))\
        .batch(128)


def dataset_api():
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32).repeat()

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(32).repeat()

    model.fit(dataset, epochs=10, steps_per_epoch=30,
              validation_data=val_dataset,
              validation_steps=3)


# ***********************CALLBACKS*************************
callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    # Write TensorBoard logs to `./logs` directory
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))
# You can write your own custom callback, or use the built-in tf.keras.callbacks that include:
#
# tf.keras.callbacks.ModelCheckpoint: Save checkpoints of your model at regular intervals.
# tf.keras.callbacks.LearningRateScheduler: Dynamically change the learning rate.
# tf.keras.callbacks.EarlyStopping: Interrupt training when validation performance has stopped improving.
# tf.keras.callbacks.TensorBoard: Monitor the model's behavior using TensorBoard.


