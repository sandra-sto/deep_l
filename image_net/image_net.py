import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50


vgg_model = vgg16.VGG16(weights='image_net')
