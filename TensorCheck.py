# Hello, GPU???

import tensorflow as tf
print("TF Version:", tf.version.VERSION)
from tensorflow import keras
tf.config.list_physical_devices('GPU')