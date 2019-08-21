from datetime import time

from tensorflow.python.keras.callbacks import TensorBoard

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

    def train(self, model, data, labels, validation_split):
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        model.fit(data,
                  labels,
                  batch_size=64,
                  epochs=5,
                  validation_split=validation_split
                  )

    def train_generator(self, model, tensorboard, train_generator, validation_generator, train_step, val_step):
        model.fit_generator(
            generator=train_generator,
            epochs=1,
            steps_per_epoch=train_step,
            validation_data=validation_generator,
            validation_steps=val_step,
            callbacks=[tensorboard])

    def compile_model(self, model, loss, optimizer, metrics):
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
        return model