# Sparse encoder generator. Saves the model so that it can be used in the AD classes.
# Richard Masson
# Created on 17th Feb 2022.
print("GENERATING SPARSE ENCODER FOR THE BRAIN SCANS")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#from nibabel import test
import LabelReader as lr
import NIFTI_Engine as ne
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt
import random
import datetime
from collections import Counter
import pandas as pd
from sklearn.model_selection import KFold
import sys

def buildEncoder(width, height, depth, compression=2):
    enc_dim = int(400/compression)
    
    inputs = keras.Input((None,width, height, depth, 1))
    
    enc = layers.Conv3DLayer(filters=64, kernel_size=5, stride=2, activation="relu")(inputs)
    enc = layers.Conv3DLayer(filters=128, kernel_size=5, stride=2, activation="relu")(inputs)
    enc = layers.Conv3DLayer(filters=128, kernel_size=5, stride=2, activation="relu")(inputs)
    # reshape here? enc = reshape(incoming=enc, shape=(-1,32*4*4*7))
    enc = layers.Reshape((-1, 32*4*4*7))
    enc = layers.Dense(units=enc_dim, activation="sigmoid", activity_regularizer=regularizers.l1(10e-5))(enc)
    
    dec = keras.Input((None,enc_dim))(enc)
    dec = keras.Dense(units=32*4*4*7, activation="sigmoid")(dec)
    dec = layers.Reshape((-1, 32, 4, 4, 7))(dec)
    dec = layers.DeConv3DLayer(filters=128, kernel_size=4, stride=2, activation="relu")(dec)
    # ?
    
    return enc, dec

print("Powering up the Brain Encoder...")
print("First we need to construct the dataset again.")

if testing_mode:
    scale = 4 # while testing, scale down the image size by a factor of X
else:
    scale = 1 # while training, do we use the full size or not?

# ADNI dimensions (need to verify this at some point)
w = int(208/scale)
h = int(240/scale)
d = int(256/scale)

# Fetch all our seperated data
adniloc = "/scratch/mssric004/ADNI_Data"
if testing_mode:
    adniloc = "/scratch/mssric004/ADNI_Test"
    print("TEST MODE ENABLED.")
x_arr, y_arr = ne.extractADNI(w, h, d, root=adniloc)
print("Data successfully loaded in.")

force_diversity = False
if testing_mode:
    if force_diversity:
        print("ACTIVATING CLASS DIVERSITY MODE")
        mid = int(len(y_arr)/2)
        new_arr = []
        for i in range (len(y_arr)):
            if i < mid:
                new_arr.append(0)
            else:
                new_arr.append(1)
        random.shuffle(new_arr)
        y_arr = new_arr
        print("Diversified set:", y_arr)

# Split data
if testing_mode:
    if force_diversity:
        x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr, stratify=y_arr)
        print("y train:", y_train)
        print("y val:", y_val)
        if len(y_val) >= 5:
            x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.2)
        else:
            print("Dataset too small to fully stratify - temporarily bypassing that...")
            x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.2)
    else:
        x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr) # ONLY USING WHILE THE SET IS TOO SMALL FOR STRATIFICATION
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.2) # ALSO TESTING BRANCH NO STRATIFY LINE
else:
    x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr, stratify=y_arr) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.2) # 80/20 val/test, therefore 75/20/5 train/val/test.
print("Data is sorted and ready.")