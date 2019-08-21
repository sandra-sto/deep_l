import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import tensorboard
input_shape = (28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# x_train = tf.cast(x_train, tf.float32)
x_train = x_train.astype('float32')
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train, x_test = x_train /255., x_test/255.

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=input_shape, padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
# model.add(Dropout(0, 2))
model.add(Dense(10, activation=tf.nn.softmax))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=100)
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Loss', test_loss)
print('Accuracy', test_acc)