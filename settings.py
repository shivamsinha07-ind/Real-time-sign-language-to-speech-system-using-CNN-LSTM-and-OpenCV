import os
import numpy as np
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

CLASSES = np.array(['HELLO','YES','NO','HELP','PLEASE','LOVE','STOP','WATER','SORRY','BYE'])

ROOT_DATA = os.path.join('dataset')

TOTAL_SEQUENCES = 20
FRAMES_PER_SEQ = 35
FRAME_SIZE = (72,72)