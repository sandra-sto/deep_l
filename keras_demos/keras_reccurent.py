import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
import keras
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

train_labels = keras.utils.to_categorical(x_train, 10)
test_labels = keras.utils.to_categorical(x_test, 10)


model = Sequential()
model.add(LSTM(128, input_shape=x_train.shape[1:], activation='relu', return_sequences=True))
# return sequences is true if we have another recurrent layer after this

model.add(Dropout(0.2))

model.add(LSTM(128, activation=keras.activations.relu))
model.add(Dropout(0.1))

model.add(Dense(32, activation=keras.activations.relu))
model.add(Dropout(0.2))

model.add(Dense(10, activation=keras.activations.softmax))



