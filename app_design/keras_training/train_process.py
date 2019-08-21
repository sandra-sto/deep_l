import tensorflow as tf

class TrainProcess:
    def __init__(self, model_creator, train, compiler, data_generator, model_io):
        self.model_creator = model_creator
        self.train = train
        self.compiler = compiler
        self.data_generator = data_generator
        self.model_io = model_io

    def start_train_process_from_scratch(self):
        # data = self.data_generator.generate_data(250)

        (data_x, data_y) = tf.keras.datasets.mnist.load_data()
        config = {'width': 1920, 'height': 1040, 'depth': 1, 'classes': 3}
        model = self.model_creator.create_model(config)

        self.train.compile_model(model, 'categorical_crossentropy', 'adam', ['accuracy'])
        self.train.train(model, data_x, data_y, 0.2)

        # self.model_io.save_model(file)

