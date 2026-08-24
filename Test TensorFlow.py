# test_tensorflow.py

import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

print("GPU Available:", len(tf.config.list_physical_devices('GPU')))