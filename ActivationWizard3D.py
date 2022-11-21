# He finds all the activation values for us. He is the Activation Wizard (2D)
# Richard Masson
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
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

model_name = "Models/ADModel_NEO_V5-basicvalid.h5"
model = load_model(model_name)
print("Keras model loaded in. [", model_name, "]")

print("Compiling...")
optim = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optim,loss='categorical_crossentropy', metrics=['accuracy'])

# Grab that data now
print("\nExtracting data")
scale = 1
w = (int)(169/scale)
h = (int)(208/scale)
d = (int)(179/scale)
classNo = 2

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
labs = to_categorical(labels, num_classes=classNo, dtype='float32')
print("Predicting on", len(path), "images.")
print("Distribution:", Counter(labels))

print("Data obtained. Mapping to a dataset...")

x_arr = []
tp = []
kp = []
saveloc = "Activations/3D/"
for i in range(1):
    func = nib.load(path[i])
    image = np.asarray(func.get_fdata(dtype='float32'))
    image = ne.organiseADNI(image, w, h, d, strip=strip_mode)
    image = np.expand_dims(image, axis=0)
    print("For:", path[i])
    print("Class:", labels[i])
    print("Prediction:", model.predict(image))
    lab = to_categorical(labels[i])
    layername = "conv3d"
    print("Getting activations")
    activations = keract2.get_activations(model, image, layer_names=layername)
    [print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]
    
    gradmap = keract2.gen_heatmaps(activations, image)
    print("The activation map has the following shape:", gradmap[0].shape)
    grad = gradmap[0]
    
    # Transform this thing
    numer = grad - np.min(grad)
    denom = (grad.max() - grad.min()) + (1e-8)
    grad = numer / denom
    grad = (grad * 255).astype("uint8")
    
    gslice = grad[:,:,80]
    plt.imshow(gslice)
    slicename = saveloc + "image" +str(i) +"_class" +str(labels[i]) +".png"
    plt.savefig(slicename)
    new_image = nib.Nifti1Image(grad, func.affine)
    saveto = saveloc + "Full/image" +str(i) +"_class" +str(labels[i]) +".nii.gz"
    nib.save(new_image, saveto)
    print("Saved to", saveto)

keras.backend.clear_session()
print("\nAll done.")

