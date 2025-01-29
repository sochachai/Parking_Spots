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
train_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(directory=train_dir, target_size=(32,32), classes=['empty','occupied'],batch_size=10)

validation_dir = 'train_data_v3/test'
#validation_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
#    .flow_from_directory(directory=validation_dir, target_size=(224,224), classes=['empty','occupied'],batch_size=1)
validation_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(directory=validation_dir, target_size=(32,32), classes=['empty','occupied'],batch_size=10)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same',input_shape=(32,32,3)),
    MaxPool2D(pool_size=(2,2),strides=2),
    Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same'),
    MaxPool2D(pool_size=(2,2),strides=2),
    Flatten(),
    Dense(units = 2,activation = 'softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=validation_batches,epochs=10, verbose=2)
model.save('base_model.h5')
model = keras.models.load_model("base_model.h5")
# get predictions
predictions = model.predict(x = validation_batches[1], verbose = 0)
print(predictions)
print(validation_batches[1][1])