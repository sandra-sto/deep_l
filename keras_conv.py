import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

input_shape = (28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.as_type('float32')
x_train = tf.cast(x_train, tf.float32)
y_train = y_train.as_type('float32')
y_train = tf.cast(y_train, tf.float32)

x_train /= 255
y_train /= 255

model = Sequential()
model.add(Conv2D(28, kernel_size = (3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation = tf.nn.relu))
model.add(Dropout(0, 2))
model.add(Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 10)
model.evaluate(x_test, y_test)