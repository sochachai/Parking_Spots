import numpy as np
import tensorflow as tf
#print(tf.__version__)
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

#from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)

'''
train_dir = 'train_data/train'
train_generator = image_dataset_from_directory(
    train_dir,
    labels='inferred',  # Infer labels from directory structure
    label_mode='categorical',  # Use one-hot encoding for labels
    color_mode='rgb',
    image_size = (224, 224),
    batch_size = 10
)

validation_dir = 'train_data/test'
validation_generator = image_dataset_from_directory(
    validation_dir,
    labels='inferred',  # Infer labels from directory structure
    label_mode='categorical',  # Use one-hot encoding for labels
    color_mode='rgb',
    image_size = (224, 224),
    batch_size = 10
)
'''
train_dir = 'train_data_v3/train'
#train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
#    .flow_from_directory(directory=train_dir, target_size=(224,224), classes=['empty','occupied'],batch_size=1)
train_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(directory=train_dir, target_size=(224,224), classes=['empty','occupied'],batch_size=10)

validation_dir = 'train_data_v3/test'
#validation_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
#    .flow_from_directory(directory=validation_dir, target_size=(224,224), classes=['empty','occupied'],batch_size=1)
validation_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(directory=validation_dir, target_size=(224,224), classes=['empty','occupied'],batch_size=10)

'''
images, _ = next(validation_batches)
#print(images)
cv2.imshow(' ',images[0])
#print(validation_batches)
#data_iterator = validation_batches.as_numpy_iterator()
#print(next(data_iterator))
#batch = data_iterator.next()
#print(batch[1])
#cv2.imshow(' ',batch[0][0])
#cv2.imshow(' ', images)
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
'''

vgg16_model = tf.keras.applications.vgg16.VGG16()
#vgg19_model = tf.keras.applications.vgg19.VGG19(include_top=False, input_shape = (224,224,3))
vgg16_model.summary()


model = Sequential()
#input_layer = keras.layers.Input(shape=(48, 48, 3),batch_size=10)
#model.add(input_layer)
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers[:10]:
    layer.trainable = False

model.add(Dense(units=2, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=validation_batches,epochs=10, verbose=2)
model.save('my_fine_tuned_vgg16.h5')
model = keras.models.load_model("my_fine_tuned_vgg16.h5")
# get predictions
predictions = model.predict(x = validation_batches[1], verbose = 0)
print(predictions)
print(validation_batches[1][1])
