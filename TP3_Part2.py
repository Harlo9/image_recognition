#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:41:06 2024

@author: Si-Thami KHAIF / Sébastien LAMARQUE / Britney HONG
"""

# -------------------------- Partie 2 ---------------------------- 
# ------------  Implementing the Interactive Window --------------


#%%-------- Step 1 : Import libraries 

import pygame 
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from gtts import gTTS
from tempfile import TemporaryFile
from pygame import mixer
import os
import pygame.surfarray


#%% -------- Step 2 : Initialize Pygame

pygame.init() # initialize Pygame
BLACK = (0, 0, 0) # defining black color
WHITE = (255, 255, 255) # defining black color
# Set up the display
WIDTH, HEIGHT = 400, 400
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Handwritten Digit Recognition")


#%% ------- Step 3 : Set Up Drawing Parameters
# Set up drawing parameters
drawing = True  
last_pos = (0,0) 
radius = 3
 

#%% ------- Step 4 : Function to predict the image 
def predict_digit(img):
    #Import the model of the first part
    model = tf.keras.models.load_model('model_reco.h5')
    img=tf.image.resize(img,[28,28])
    img=tf.reshape(img,(1,28,28,1))
    prediction=model.predict(img)
    digit=np.argmax(prediction)
    return digit

#%% ------- Step 5 : Function to display text in the corner
def display_text(text):
    font=pygame.font.Font(None,72)
    text_surface=font.render(text,True,BLACK)
    text_rect=text_surface.get_rect(topright=(WIDTH-20,20))
    window.blit(text_surface,text_rect)


#%% --------------- Step Bonus : Text to Speech 

def text_to_speech(digit, lang='en'):
    full_text = f"The number is {digit}"
    tts = gTTS(text=full_text, lang=lang)
    temp_file = TemporaryFile()
    tts.write_to_fp(temp_file)
    temp_file.seek(0)
    mixer.init()
    mixer.music.load(temp_file)
    mixer.music.play()
    while mixer.music.get_busy():
        continue
    temp_file.close()

#%% ------ Step 6 :  Implement Drawing Functionality

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                pos = pygame.mouse.get_pos()
                pygame.draw.line(window, BLACK, last_pos, pos, radius)
                last_pos = pos

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        window.fill(WHITE)
    if keys[pygame.K_RETURN]:
       
        img = pygame.surfarray.array3d(window)
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)  # add channel dimension
                
        # Predict the digit using the CNN model
        digit = predict_digit(img)
        display_text(str(digit))
        text_to_speech(digit)
        
    pygame.display.flip()

pygame.quit()

"""

each if represent each event when the user interact with the pygame window. 
(K_space, enter bouton, and the motion of the mouse have each an event)

"""


