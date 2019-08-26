from datetime import time

from tensorflow.python.keras.callbacks import TensorBoard
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# config.gpu_options.allow_growth = True
# vgg16_front = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=imgShape)
class Train:
    def __init__(self):
        return

    # def train(self, model, train_data, labels, validation_data):
    #     tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    #     model.fit(train_data,
    #               labels,
    #               batch_size=64,
    #               epochs=5,
    #               )

    def train(self, model, data, labels, validation_split, batch_size, epochs, callbacks):
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        model.fit(data,
                  labels,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=validation_split,
                  callbacks=callbacks
                  )

    def train_generator(self, model, callbacks, train_generator, validation_generator, train_step, val_step, epochs):
        model.fit_generator(
            generator=train_generator,
            epochs=epochs,
            steps_per_epoch=train_step,
            validation_data=validation_generator,
            validation_steps=val_step,
            callbacks=callbacks)

    def compile_model(self, model, loss, optimizer, metrics):
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
        return model