class Preprocessing:
    def normalize(self, x):
        raise NotImplementedError()

    def normalize_image(self, x):
        x = x.astype('float32')
        return x/255.