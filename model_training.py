#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:43:47 2024

@author: Si-Thami KHAIF / Sébastien LAMARQUE / Britney HONG
"""
# -------------------------- Partie 1 ---------------------------- 
# --------------- Constructing the Neural Network ----------------


#%% ----------- Step 1 : Import Libraries

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


#%% ----------- Step 2 : Load the data 

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train)
print("Shape of x_train:", x_train.shape)

"""
Here we can see that the x_train matrix have coefficient beetween 
0 to 255 wich represent the color of each pixels of the image

"""

i = 0  

for i in range(0, 9):
    plt.imshow(x_train[i], cmap='binary')
    plt.title(f"Label: {y_train[i]}")  
    plt.show()


#%% ----------- Step 3 : Preprocess the Data

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10) # specify num_classes
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10) # specify num_classes


#%% --------- Step 4 : Define model parameters

model = models.Sequential()

# Add Convolutional Layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output
model.add(layers.Flatten())

# Add Fully Connected Layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Adding Dropout layer for regularization
# Output layer with softmax activation for multi-class classification
model.add(layers.Dense(10, activation='softmax'))  


#%% --------- Step 5 : Compile the model 

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#%% --------- Step 6 : Train the model 

history = model.fit(x_train, y_train, epochs=5, batch_size=64, 
                    validation_data=(x_test, y_test))


#%% --------- Step 7 : Evaluate the model

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')


#%% --------- Step 8 : Save our model 

model.save("model_reco.h5") 










