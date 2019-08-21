from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

class ImageGenerator:
    def __init__(self):

        self.train_data_gen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    def generate_data(self, batch_size):
        data = self.train_data_gen.flow_from_directory(
            'data/train',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')

        return data

    # validation_generator = test_datagen.flow_from_directory(
    #     'data/validation',
    #     target_size=(150, 150),
    #     batch_size=batch_size,
    #     class_mode='binary')
