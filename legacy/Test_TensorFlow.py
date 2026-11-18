# test_tensorflow.py

try:
    import tensorflow as tf
except ModuleNotFoundError:
    print("TensorFlow is not installed in the current environment.")
    print("Install it with: pip install tensorflow-cpu")
    raise SystemExit(1)

print("TensorFlow Version:", tf.__version__)

print("GPU Available:", len(tf.config.list_physical_devices('GPU')))