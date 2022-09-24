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
from collections import Counter
print("Imports working.")

# Flags
display_mode = True # Print out all the weight info
single = True
slicer = False
memory_mode = False # Print out memory summaries
strip_mode = False

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
print("Keras model loaded in.")

#GPUtil.showUtilization()
weights=model.get_weights()
#print("No. of weights elements =", len(weights))
#for i in range(len(weights)):
#    print("Weight ", (i+1), ": ", weights[i].shape, sep='')

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()
wi = [] # Weight info

for name, weight in zip(names, weights):
    string = name +" " +str(weight.shape)
    wi.append(string)
if display_mode:
    for element in wi:
        print(element)

# Parameters for recreating model
class_no = 2 # Need to hardcode this for now

# Recreate model in Torch
class TorchBrain(nn.Module):
    def __init__(self):
        super(TorchBrain, self).__init__()
        # Input size is 208, 240, 256, 1
        # Step 1: Conv3D, 32 filters, 3x kernel, relu (3, 3, 3, 1, 32)
        self.conv = nn.Conv3d(1, 32, 3, padding='valid')
        self.relu = nn.LeakyReLU()
        # Step 2: 10x10, stride 10 max pooling
        self.pool = nn.MaxPool3d(10, stride=10, padding='valid')

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

class TorchBrain2D(nn.Module): # For more complex model
    def __init__(self):
        super(TorchBrain2D, self).__init__()
        # Input size is 208, 240, 256, 1
        # Step 1: Conv3D, 32 filters, 3x kernel, relu (3, 3, 3, 1, 32)
        self.conv = nn.Conv2d(1, 32, 3, padding='valid')
        self.relu = nn.LeakyReLU()
        # Step 2: 10x10, stride 10 max pooling
        self.pool = nn.MaxPool2d(5, stride=5)

        # Step 3: Flatten and dense layer
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(42240, 128) #???

        # Step 5: Final Dense layer, softmax
        self.dense2 = nn.Linear(128, class_no)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        print("Forward pass...")
        print("Input shape:", x.shape)
        out = self.conv(x)
        print("After Conv (1):", out.shape)
        out = self.relu(out)
        print("After Relu:", out.shape)
        out = self.pool(out)
        print("After Pooling (1):", out.shape)
        out = self.flatten(out)
        print("After flattening:", out.shape)
        out = self.dense1(out)
        print("After Dense (1):", out.shape)
        out = self.dense2(out)
        print("After Dense (2):", out.shape)
        out = self.softmax(out)
        print("After Softmax:", out.shape)
        
        return out

# Generate torch model
torchy = TorchBrain2D()
print("Torch model loaded in.")
if display_mode:
    print("\n", torchy, "\n", sep='')
    print("KERAS SUMMARY:")
    model.summary()

# Print out all the weight shapes so we can get the transposition right
# torch weights, bias, running_mean, running_var corresponds to keras gamma, beta, moving mean, moving average
if display_mode:
    print("\n")
    print(wi[0], "| Torch:", torchy.conv.weight.shape)
    print(wi[1], "| Torch:", torchy.conv.bias.shape)
    print(wi[2], "| Torch:", torchy.dense1.weight.shape)
    print(wi[3], "| Torch:", torchy.dense1.bias.shape)
    print(wi[4], "| Torch:", torchy.dense2.weight.shape)
    print(wi[5], "| Torch:", torchy.dense2.bias.shape)

# Copy over all the weights
# Correct transposing is needed here.
torchy.conv.weight.data = torch.from_numpy(np.transpose(weights[0]))
torchy.conv.bias.data = torch.from_numpy(weights[1])
torchy.dense1.weight.data = torch.from_numpy(np.transpose(weights[2]))
torchy.dense1.bias.data = torch.from_numpy(weights[3])
torchy.dense2.weight.data = torch.from_numpy(np.transpose(weights[4]))
torchy.dense2.bias.data = torch.from_numpy(weights[5])

# GPU MODE ENGAGE
#print(torch.cuda.get_device_name(0))
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
torchy.to(device=device)
#del model
#keras.backend.clear_session()
#GPUtil.showUtilization()

# Grab that data now
print("\nExtracting data")
scale = 1
w = (int)(169/scale)
h = (int)(208/scale)
d = (int)(179/scale)

if single:
    if slicer:
        imgname = "Directories\\test_single_adni_2_images.txt"
        labname = "Directories\\test_single_adni_2_labels.txt"
    else:
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
if slicer:
    temp = path[0]
    path = os.listdir(path[0])
    for i in range(0, len(path)):
        path[i] = os.path.join(temp, path[i])
label_file = open(labname, 'r')
labels = label_file.read()
labels = labels.split("\n")
labels = [ int(i) for i in labels]
label_file.close()
#labels = to_categorical(labels, num_classes=class_no, dtype='float32')
#print(path)
print("Predicting on", len(path), "images.")
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
    return tf.py_function(load_val, [file], [np.float32])
'''
'''
print("Data obtained. Mapping to a dataset...")

# Convert to a dataset
x = tf.data.Dataset.from_tensor_slices((path))

data = (
    x.shuffle(len(x))
    .map(load_img_wrapper)
    .batch(1)
)
'''
'''
tensor_x = torch.Tensor(path)
tensor_x = tensor_x.to(device=device)
data = TensorDataset(tensor_x)
data = data.map(load_img_wrapper)
loader = DataLoader(data, batch_size=1, shuffle=False)
'''
'''
tensor_x = torch.Tensor(x) # transform to torch tensor
tensor_x = tensor_x.to(device=device)
print("Type [tensor_x]:", type(tensor_x))
tensor_x = torch.permute(tensor_x, (0, 4, 1, 2, 3))
data = TensorDataset(tensor_x) # create your datset

data_loader = DataLoader(data, batch_size= 1, shuffle=False) # Set shuffle to true when we want to look at more than just the first instance.
print("Type [data_loader]:", data_loader)
'''
# Which layer are we observing?
#layerchoice = "dense2"
#print("Layer chosen:", layerchoice, "- Shape:", torchy.dense2.weight.data.shape)
#print("Examples:", torchy.dense2.weight.data[:5])

# Cam injection
print("Done. Attempting injection...")
#cam_model = medcam.inject(torchy, output_dir='Grad-Maps-2D', backend='gcam', layer='relu', label='best', save_maps=True) # Removed label = 'best'
print("Injection successful.")
'''
sing = np.asarray(nib.load(path[0]).get_fdata(dtype='float32'))
image = ne.organiseADNI(sing, w, h, d, strip=strip_mode)
image = image[:,:,modelnum]

from PIL import Image
img_arr = np.squeeze(image, axis=2)
im = Image.fromarray((img_arr * 255).astype(np.uint8))
print("shape:", img_arr.shape)
im.convert('L')
im.save("Grad-Maps-2D/reference.jpg")
'''
'''
import pydicom
dicom = pydicom.read_file(path[modelnum])
arr = dicom.pixel_array
cv2.imwrite("Grad-Maps-2D/reference_pre.jpg", arr)
#image = ne.normalize_per(arr)
#cv2.imwrite("Grad-Maps-2D/reference_normed.jpg", image)
image = ne.organiseSlice(arr, w, h, strip=True)
image = image.astype(float)
cv2.imwrite("Grad-Maps-2D/reference_cropped.jpg", image)
print("Input shape is:", image.shape)
'''
'''
image = np.expand_dims(image, axis=0)
print("Input shape is:", image.shape)
x = torch.Tensor(image)
x = x.to(device=device)
print("Permuting...")
x = torch.permute(x, (0, 3, 1, 2))
print("Shape after permutation:", x.shape)
pred = cam_model(x)
pred_read = pred.detach().cpu().numpy()
#print(pred_read.shape)
print("Prediction:", pred_read, "(actual:", labels[0], ")")
'''

x_arr = []
tp = []
kp = []
for i in range (len(path)):
    # Incredibly dirty way of preparing this data
    image = np.asarray(nib.load(path[i]).get_fdata(dtype='float32'))
    image = ne.organiseADNI(image, w, h, d, strip=strip_mode)
    image = image[:,:,modelnum]
    image = np.expand_dims(image, axis=0)
    #print("Sending in an image of shape", image.shape)
    x = torch.Tensor(image)
    x = x.to(device=device)
    x = torch.permute(x, (0, 3, 1, 2))
    #x_arr.append(x) # ?
    #pred = cam_model(x)
    pred = torchy(x)
    pred_read = pred.detach().cpu().numpy()
    preds = np.argmax(pred_read, axis=1)
    print("\nTorch prediction:", pred_read[0], "| (", preds[0], "vs. actual:", labels[i], ")")
    tp.append(preds[0])
    kerpred = model.predict(image)
    kerpreddy = np.argmax(kerpred, axis=1)
    print("Keras prediction:", kerpred[0], "| (", kerpreddy[0], "vs. actual:", labels[i], ")")
    kp.append(kerpreddy[0])

print("Torch:", tp)
print("Actual:", labels)
corr = 0
total = 0
for i in range(0, len(labels)):
    if tp[i] == labels[i]:
        corr += 1
    total += 1
acc = (corr/total)*100
print("Rough acc: ", acc, "%", sep='')

#print("Image shape:", x.shape)
#print("Image shape:", x.shape)
keras.backend.clear_session()
print("\nAll done.")
