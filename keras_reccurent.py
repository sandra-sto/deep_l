import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

model = Sequential()
model.add(LSTM(128, input_shape=x_train.shape[1:], activation='relu', return_sequences=True))
# return sequences is true if we have another reccurent layer after this

model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



