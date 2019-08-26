from app_design.model_creator.model_creator import ModelCreator
import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, InputLayer
# batch normalizacija samo  na treningu
class HardCodedModelCreator(ModelCreator):
    def create_model(self, config):
        width = config['width']
        height = config['height']
        depth = config['depth']
        classes = config['classes']

        input_shape = (height, width, depth)
        chanDim = -1

        model = Sequential(name = "MiniVGG")
        with tf.name_scope('Conv_Block_1'):
            # model.add(InputLayer(input_shape=input_shape))
            model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
            model.add(Activation(tf.keras.layers.Lambda(lambda t: tf.nn.crelu(t))))
            model.add(BatchNormalization(axis = chanDim))
            model.add(Conv2D(32, (3, 3), padding="same"))
            model.add(Activation(tf.keras.layers.Lambda(lambda t: tf.nn.crelu(t))))
            model.add(BatchNormalization(axis = chanDim))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            # define the model input

        # second (CONV => RELU) * 2 => POOL layer set
        with tf.name_scope('Conv_Block_2'):
            model.add(Conv2D(64, (3, 3), padding="same"))
            model.add(Activation(tf.keras.layers.Lambda(lambda t: tf.nn.crelu(t))))
            model.add(BatchNormalization(axis = chanDim))
            model.add(Conv2D(64, (3, 3), padding="same"))
            model.add(Activation(tf.keras.layers.Lambda(lambda t: tf.nn.crelu(t))))
            model.add(BatchNormalization(axis = chanDim))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        with tf.name_scope('Dense_Block'):
            model.add(Flatten())
            model.add(Dense(512, kernel_initializer='orthogonal'))
            model.add(Activation(tf.keras.layers.Lambda(lambda t: tf.nn.crelu(t))))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(tf.keras.layers.Dense(classes, activation='softmax'))

        model.summary()

        return model

