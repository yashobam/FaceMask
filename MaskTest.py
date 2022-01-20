# %% Imports
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import time

# %% Import model
model = tf.keras.models.load_model('Saved_Model/my_model')

# %% Launch Webcam with model
#testImage = "https://storage.googleapis.com/mask_data_photos/TestImages/Screen%20Shot%202021-07-02%20at%201.39.12%20PM.png"
#testImage_path = tf.keras.utils.get_file('hihihihi12', origin=testImage)
batch_size = 128
img_height = 180
img_width = 180
class_names = ['no', 'yes']
testImage_path="IMG_6720.jpg"
testTimer = time.time()
img = keras.preprocessing.image.load_img(testImage_path, target_size=(img_height, img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(
    class_names[np.argmax(score)], 100 * np.max(score)))

print(time.time() - testTimer)
