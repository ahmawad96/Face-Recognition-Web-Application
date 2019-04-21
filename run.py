import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from fr_utils import *
from model import *
from FaceProcessing import *

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2


# Creating a python dictionary to represent our database.
database = {}
def login():
    global database
    print("enter your name:")
    name=str(input())
    if name in database:
        user_face=DetectFace()
        user_face=prepare(user_face)
        encoding = img_to_encoding(user_face, FRmodel)
        dist = np.linalg.norm(encoding-database[name])
        if dist < 0.7:
            print("It's " + str(name) + ", Login successful!")
            login = True
        else:
            print("It's not " + str(name) + ", Login failed... ")
            login = False
    else:
        print("Name not found!")

    return dist, login

def signup():
    global database
    print("enter your name:")
    name=str(input())
    user_face=DetectFace()
    user_face=prepare(user_face)
    database[name] = img_to_encoding(user_face, FRmodel)
    print("User:"+ name +" added to database!")
    #print(database[name])

##################################################


FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
print("Loading weights!")
load_weights_from_FaceNet(FRmodel)
print("Finished loading weights!")


while(True):
    print("type your request: signup or login?")
    request=str(input())
    if(request=="signup"):
        signup()

    elif request=="login":
        login()




# Let's test if it can verify me!
#print("Verifying")
#verify("camera_0.jpg", "Ahmed", database, FRmodel)
