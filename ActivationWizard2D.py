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
# Just pick a single model for now
modelnum = 58
model_name = "Models/2DSlice_V2-prio/model-"+str(modelnum)+".h5"
model = load_model(model_name)
print("Keras model loaded in. [", model_name, "]")

print("Compiling...")
optim = keras.optimizers.Adam(learning_rate=0.001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later
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
print("Data obtained. Mapping to a dataset...")

x_arr = []
tp = []
kp = []
#for i in range (len(path)):
for i in range(1):
    # Incredibly dirty way of preparing this data
    image = np.asarray(nib.load(path[i]).get_fdata(dtype='float32'))
    image = ne.organiseADNI(image, w, h, d, strip=strip_mode)
    image = image[:,:,modelnum]
    plt.imshow(image)
    plt.savefig("Activations\original.png")
    image = np.expand_dims(image, axis=0)
    # Here goes nothing
    layername = "conv2d_2"
    print("Getting activations for layer:", layername)
    activations = get_activations(model, image, auto_compile=True)
    conv = get_activations(model, image, layer_names=layername)
    #activations = get_activations(model, image, auto_compile=True, layer_names=layername)
    # Print activations shapes
    [print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]
    saving = True
    if saving:
        print("Saving heatmap to /Activations/")
    else:    
        print("Displaying conv heatmap:")
    keract.display_activations(activations, save=saving, directory='Activations')
    #keract.display_heatmaps(conv, image, directory='Activations', save=True, fix=True, merge_filters=True)
    '''
    kerpred = model.predict(image)
    kerpreddy = np.argmax(kerpred, axis=1)
    print("Keras prediction:", kerpred[0], "| (", kerpreddy[0], "vs. actual:", labels[i], ")")
    kp.append(kerpreddy[0])
    '''

keras.backend.clear_session()
print("\nAll done.")


