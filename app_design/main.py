from rope.base.oi.type_hinting.evaluate import Compiler

from app_design.data_generators.image_generator import ImageGenerator
from app_design.keras_training.keras_train import Train
from app_design.keras_training.train_process import TrainProcess
from app_design.model_creator.hard_coded_model_creator import HardCodedModelCreator
from app_design.model_creator.model_creator import ModelCreator
from app_design.models_io.models_io import ModelIO

model_creator = HardCodedModelCreator()
train = Train()
compiler = Compiler()
data_generator = ImageGenerator()
model_io = ModelIO()
train_process = TrainProcess(model_creator, train, compiler, data_generator, model_io)
train_process.start_train_process_from_scratch()

