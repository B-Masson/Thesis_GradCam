# Can you solve all my problems, O Activation Wizard?
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print("Imports working.")

# Flags
display_mode = True # Print out all the weight info
single = True
memory_mode = False # Print out memory summaries
strip_mode = False
basic_mode = True
saving = True

# Memory setup
if memory_mode:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    GPUtil.showUtilization()

priority_slices = [56, 57, 58, 64, 75, 85, 88, 89, 96]
slice_range = np.arange(50, 100)
# Just pick a single model for now
#modelnum = 58
#model_name = "Models/2DSlice_V2-prio/model-"+str(modelnum)+".h5"
model_dir = "Models/2DSlice_V1.5-entire/model-"
#model = load_model(model_name)
#print("Keras model loaded in. [", model_name, "]")

#print("Compiling...")
#optim = keras.optimizers.Adam(learning_rate=0.001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later
#model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

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
#labels = to_categorical(labels, num_classes=class_no, dtype='float32')
#print(path)
print("Predicting on", len(path), "images.")
#print(path)
print("Distribution:", Counter(labels))
#GPUtil.showUtilization()
'''
# Dataset loaders
def load_img(file): # NO AUG, NO LABEL
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    nifti = ne.organiseADNI(nifti, w, h, d, strip=strip_mode)
    nifti = tf.convert_to_tensor(nifti, np.float32)
    return nifti

def load_img_wrapper(file):
    return tf.py_function(load_img, [file], [np.float32])
'''

# Incredibly dirty way of preparing this data
image_raw = np.asarray(nib.load(path[0]).get_fdata(dtype='float32'))
image_raw = ne.organiseADNI(image_raw, w, h, d, strip=strip_mode)
lab = to_categorical(labels[0])
print("Data obtained. Moving on to models...")

# Vars
actloc = "Activations/2D/"
gradloc = "Gradients/2D/"
depth = 2
actvolume = np.zeros((167, 206, depth))
gradvolume = []
eps=1e-8

#for slicenum in slice_range:
for i in range(depth):
    slicenum = slice_range[i]
    model_name = model_dir + str(slicenum)
    model = load_model(model_name)
    print("Keras model loaded in. [", model_name, "]")
    #model.summary()
    #print("Compiling...")
    optim = keras.optimizers.Adam(learning_rate=0.001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    # Prep slice
    image = image_raw[:,:,slicenum]
    image = np.expand_dims(image, axis=0)
    #print("Image shape:", image.shape)
    # Here goes nothing
    if i == 0:
        layername = "conv2d"
    else:
        layername = "conv2d_"+str(i)
    print("Getting activations for layer:", layername)
    activations = keract2.get_activations(model, image, layer_names=layername)
    # Act heatmap
    #print("Getting heatmap")
    keract.display_heatmaps(activations, image, directory='.', save=False, fix=True, merge_filters=True)
    gradmap = keract2.gen_heatmaps(activations, image)
    print("The activation map has the following shape:", gradmap[0].shape)
    grad = gradmap[0]
    # Transform this thing
    numer = grad - np.min(grad)
    denom = (grad.max() - grad.min()) + eps
    grad = numer / denom
    grad = (grad * 255).astype("uint8")
    print("Array shape:", grad.shape)
    # Save slice
    print("Displaying...")
    plt.imshow(grad)
    slicename = actloc + "class" +str(labels[0]) +"_slice" +str(slicenum) +".png"
    plt.savefig(slicename)
    plt.clf()
    actvolume[:,:,i] = grad
    '''
    kerpred = model.predict(image)
    kerpreddy = np.argmax(kerpred, axis=1)
    print("Keras prediction:", kerpred[0], "| (", kerpreddy[0], "vs. actual:", labels[i], ")")
    kp.append(kerpreddy[0])
    '''

#np.swapaxes(actvolume, axis1, axis2)
print("Activation volume shape:", actvolume.shape)
keras.backend.clear_session()
print("\nAll done.")


