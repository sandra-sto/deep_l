import matplotlib.pyplot as plt
import keras
(train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_x[i])
plt.show()

train_y = keras.utils.to_categorical(train_y)
test_y = keras.utils.to_categorical(test_y)

