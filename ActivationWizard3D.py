# Can you solve all my problems, O Activation Wizard?
# Had to compile it to use sparse loss, for some reason.
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
single = False
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
optim = keras.optimizers.Adam(learning_rate=0.001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later
#model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
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
saveloc = "Activations/Manual/"
#for i in range (len(path)):
for i in range(1):
    # Incredibly dirty way of preparing this data
    func = nib.load(path[i])
    image = np.asarray(func.get_fdata(dtype='float32'))
    image = ne.organiseADNI(image, w, h, d, strip=strip_mode)
    image = np.expand_dims(image, axis=0)
    # Here goes nothing
    lab = to_categorical(labels[i])
    layername = "conv3d"
    print("Getting activations")
    activations = keract2.get_activations(model, image, layer_names='conv3d')
    [print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]
    
    gradmap = keract2.gen_heatmaps(activations, image)
    print("The activation map has the following shape:", gradmap[0].shape)
    grad = gradmap[0]
    
    # Transform this thing
    numer = grad - np.min(grad)
    denom = (grad.max() - grad.min()) + eps
    grad = numer / denom
    grad = (grad * 255).astype("uint8")
    
    gslice = grad[:,:,80]
    plt.imshow(gslice)
    slicename = saveloc + "image" +str(i) +"_class" +str(labels[i]) +".png"
    plt.savefig(slicename)
    #plt.show()
    new_image = nib.Nifti1Image(grad, func.affine)
    nib.save(new_image, saveloc + "NIFTI/image" +str(i) +"_class" +str(labels[i]) +".nii.gz")
    #conv = get_activations(model, image, layer_names=layername)
    #grads = keract.get_gradients_of_activations(model, image, [1], layer_names=layername, output_format='simple')
    #print("Grads")
    #[print(k, '->', v.shape, '- Numpy array') for (k, v) in grads.items()]


keras.backend.clear_session()
print("\nAll done.")

