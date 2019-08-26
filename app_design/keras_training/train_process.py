import time

import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical

class TrainProcess:
    def __init__(self, model_creator, train, compiler, data_generator, model_io):
        self.model_creator = model_creator
        self.train = train
        self.compiler = compiler
        self.data_generator = data_generator
        self.model_io = model_io

    def prepare_data(self):
        width, height, depth = 28, 28, 1
        (data_x, data_y) , _= tf.keras.datasets.mnist.load_data()
        data_x = data_x.reshape(data_x.shape[0], height, width, depth)

        data_y = to_categorical(data_y)
        config = {'width': width, 'height': height, 'depth': depth, 'classes': 10}

        return data_x, data_y, config

    def start_train_process_from_scratch(self):
        # data = self.data_generator.generate_data(250)
        data_x, data_y, config = self.prepare_data()
        early_stop = tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs{}'.format(time))
        callbacks = [early_stop, tensorboard]

        model = self.model_creator.create_model(config)

        self.train.compile_model(model, 'categorical_crossentropy', 'adam', ['accuracy'])
        self.train.train(model, data_x, data_y, 0.2, 24, 10, callbacks)

        # self.model_io.save_model(file)

