# Can you solve all my problems, O Activation Wizard?
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
memory_mode = False # Print out memory summaries
strip_mode = False
basic_mode = True

# Memory setup
if memory_mode:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    GPUtil.showUtilization()

# Oh we're doing ALL OF THEM
slicerange = np.arange(50,181)

# Need to load in a base model, then can swap out the weights per slice
model_name = "Models/2DSlice_V2-prio/model-"+str(modelnum)+".h5"
model = load_model(model_name)
print("Keras model loaded in. [", model_name, "]")
print("Compiling...")
optim = keras.optimizers.Adam(learning_rate=0.001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

# Grab that data now
print("\nExtracting data")
scale = 1
w = (int)(169/scale)
h = (int)(208/scale)
d = (int)(179/scale)

imgname = "Directories\\single.txt"
labname = "Directories\\single-label.txt"
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
labs = to_categorical(labels, num_classes=2, dtype='float32')
#print(path)
print("Predicting on", path[0])
#print(path)
print("Distribution:", Counter(labels))
#GPUtil.showUtilization()

# The IMAGE
func = nib.load(path[0])
image = np.asarray(func).get_fdata(dtype='float32')
image = ne.organiseADNI(image, w, h, d, strip=strip_mode)
fullim = []

saveloc = "Activations/2D/"
# Loop through every slice
for sliceno in slicerange:
    # Prep model
    weightloc = "need to do"
    model.load_weights(weightloc)
    # Prep slice
    slicey = image[:,:,sliceno]
    slicey = np.expand_dims(slicey, axis=0)
    # Here goes nothing
    layername = "conv2d_2"
    print("Getting activations for layer:", layername)
    activations = keract2.get_activations(model, image, layer_names=layername)
    gradmap = keract2.gen_heatmaps(activations, slicey)
    print("The activation map has the following shape:", gradmap[0].shape)
    grad = gradmap[0]
    # Transform this thing
    numer = grad - np.min(grad)
    denom = (grad.max() - grad.min()) + eps
    grad = numer / denom
    grad = (grad * 255).astype("uint8")
    # Save 2D images
    plt.imshow(grad)
    slicename = saveloc + "slice" +str(sliceno) +"_class" +str(labels[0]) +".png"
    plt.savefig(slicename)
    # Try to reconstruct it
    fullim.append(grad, axis=-1)

print("Full image shape:", fullim.shape)
new_image = nib.Nifti1Image(fullim, func.affine)
nib.save(new_image, saveloc + "NIFTI/fullim" +"_class" +str(labels[i]) +".nii.gz")
keras.backend.clear_session()
print("\nAll done.")


