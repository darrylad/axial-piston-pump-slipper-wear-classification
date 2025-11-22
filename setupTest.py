import tensorflow as tf

print("TF version:", tf.__version__)
print("All devices:", tf.config.list_physical_devices())
print("GPUs:", tf.config.list_physical_devices("GPU"))
