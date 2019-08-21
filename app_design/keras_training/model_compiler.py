from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizers import Adam
class ModelCompiler:
    def __init__(self):
        return

    def compile_model(self, model, loss, optimizer, metrics):
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
        return model