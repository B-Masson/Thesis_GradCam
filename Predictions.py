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
from medcam import medcam
import NIFTI_Engine as ne
import nibabel as nib
import matplotlib.pyplot as plt
import GPUtil
import sys
print("Imports working.")

# Load Keras model
model_name = "ADModel_TWIN_v1.5_strip.h5"
model = load_model(model_name)
weights=model.get_weights()
print("Keras model loaded in.")
class_no = 2
strip_mode = False

# Load Torch model
# Recreate model in Torch
class TorchBrain(nn.Module):
    def __init__(self):
        super(TorchBrain, self).__init__()
        # Input size is 208, 240, 256, 1
        # Step 1: Conv3D, 32 filters, 3x kernel, relu (3, 3, 3, 1, 32)
        self.conv = nn.Conv3d(1, 32, 3, padding='valid')
        self.relu = nn.LeakyReLU()
        # Step 2: 10x10, stride 10 max pooling
        self.pool = nn.MaxPool3d(10, stride=10)

        # Step 3: Flatten and dense layer
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(368000, 128) #???

        # Step 5: Final Dense layer, softmax
        self.dense2 = nn.Linear(128, class_no)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        print("Forward pass...")
        #print("Input shape:", x.shape)
        out = self.conv(x)
        #print("After Conv (1):", out.shape)
        out = self.relu(out)
        #print("After Relu:", out.shape)
        out = self.pool(out)
        #print("After Pooling (1):", out.shape)
        out = self.flatten(out)
        #print("After flattening:", out.shape)
        out = self.dense1(out)
        #print("After Dense (1):", out.shape)
        out = self.dense2(out)
        #print("After Dense (2):")
        out = self.softmax(out)
        #print("After Softmax:", out.shape)
        
        return out

# Generate torch model
torchy = TorchBrain()
torchy.conv.weight.data = torch.from_numpy(np.transpose(weights[0]))
torchy.conv.bias.data = torch.from_numpy(weights[1])
torchy.dense1.weight.data = torch.from_numpy(np.transpose(weights[2]))
torchy.dense1.bias.data = torch.from_numpy(weights[3])
torchy.dense2.weight.data = torch.from_numpy(np.transpose(weights[4]))
torchy.dense2.bias.data = torch.from_numpy(weights[5])
device = 'cpu'
torchy.to(device=device)
print("Torch model loaded in.")

# Get paths
imgname = "Directories\\test_tiny_adni_1_images.txt"
labname = "Directories\\test_tiny_adni_1_labels.txt"
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
labels = to_categorical(labels, num_classes=class_no, dtype='float32')

# Prep data
scale = 1
w = (int)(208/scale)
h = (int)(240/scale)
d = (int)(256/scale)

image = np.asarray(nib.load(path[0]).get_fdata(dtype='float32'))
image = ne.organiseADNI(image, w, h, d, strip=strip_mode)
image = np.expand_dims(image, axis=0)
print("Data prepped.")

# Predict
print("Making predictions...")
predk = model.predict(image)
x = torch.Tensor(image)
x = torch.permute(x, (0, 4, 1, 2, 3))
predt = torchy(x)
print("Keras prediction:", np.around(predk,3))
print("Torch prediction:", predt)
print("Vs. actual:", labels[0])