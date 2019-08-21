from app_design.model_creator.model_creator import ModelCreator


class ConfigModelCreator(ModelCreator):
    def create_model(self, config):
        raise NotImplementedError()
