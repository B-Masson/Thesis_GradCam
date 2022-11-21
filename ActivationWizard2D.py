# He finds all the activation values for us. He is the Activation Wizard (2D)
# Richard Masson
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import cv2
import NIFTI_Engine as ne
import nibabel as nib
import matplotlib.pyplot as plt
import GPUtil
import sys
from collections import Counter
import keract
from keract import get_activations
import KeractAlt as keract2
print("Imports working.")

# Flags
display_mode = True # Print out all the weight info
single = True
memory_mode = False # Print out memory summaries
strip_mode = False
basic_mode = True

# Memory setup
if memory_mode:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    GPUtil.showUtilization()

priority_slices = [56, 57, 58, 64, 75, 85, 88, 89, 96]
slice_range = np.arange(50, 100)
model_dir = "Models/2DSlice_V1.5-entire/model-"
model = load_model(model_name)
print("Keras model loaded in. [", model_name, "]")

print("Compiling...")
optim = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

# Grab that data now
print("\nExtracting data")
scale = 1
w = (int)(169/scale)
h = (int)(208/scale)
d = (int)(179/scale)

if single:
    imgname = "Directories\\single.txt"
    labname = "Directories\\single-label.txt"
else:
    imgname = "Directories\\test_adni_2_trimmed_images.txt"
    labname = "Directories\\test_adni_2_trimmed_labels.txt"
print("Reading from", imgname, "and", labname)
path_file = open(imgname, "r")
path = path_file.read()
path = path.split("\n")
path_file.close()
label_file = open(labname, 'r')
labels = label_file.read()
labels = labels.split("\n")
labels = [ int(i) for i in labels]
label_file.close()

print("Predicting on", len(path), "images.")
print("Distribution:", Counter(labels))

print("Data obtained. Mapping to a dataset...")

x_arr = []
tp = []
kp = []
saveloc = "Activations/2D/"
for i in range (len(path)):
    image = np.asarray(nib.load(path[i]).get_fdata(dtype='float32'))
    image = ne.organiseADNI(image, w, h, d, strip=strip_mode)
    image = image[:,:,modelnum]
    image = np.expand_dims(image, axis=0)
    lab = to_categorical(labels[i])
    layername = "conv2d_2"
    print("Getting activations for layer:", layername)
    activations = keract2.get_activations(model, image, layer_names=layername)
    [print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]
    keract.display_activations(activations, save=saving, directory='Activations')

keras.backend.clear_session()
print("\nAll done.")


