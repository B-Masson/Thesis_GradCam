from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from keras.models import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import cv2
from medcam import medcam
import NIFTI_Engine as ne
import matplotlib.pyplot as plt
print("Imports working.")

model_name = "oasis_weights.h5"
model = load_model(model_name)
weights=model.get_weights()
#print("No. of weights elements =", len(weights))
#for i in range(len(weights)):
#    print("Weight ", (i+1), ": ", weights[i].shape, sep='')

print("\n")    
names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()
wi = [] # Weight info

for name, weight in zip(names, weights):
    string = name +" " +str(weight.shape)
    wi.append(string)
for element in wi:
    print(element)

# Parameters for recreating model
class_no = 2 # Need to hardcode this for now

# Recreate model in Torch
class TorchBrain(nn.Module):
    def __init__(self):
        super(TorchBrain, self).__init__()
        # Input size is 208, 240, 256, 1
        # Step 1: Conv3D, 32 filters, 5x kernel, relu (5, 5, 5, 1, 32)
        self.conv1 = nn.Conv3d(1, 32, 5)
        self.relu = nn.LeakyReLU()
        self.pool1 = nn.MaxPool3d(2)
        self.batch1 = nn.BatchNorm3d(32, eps=1e-3)
        #Step 2: Same, but 64 filters
        self.conv2 = nn.Conv3d(32, 64, 5)
        # Relu go here
        self.pool2 = nn.MaxPool3d(2)
        self.batch2 = nn.BatchNorm3d(64, eps=1e-3)
        # Step 4: Global ave, 128 Dense, Dropout 0.2
        self.pool3 = nn.AvgPool3d(64)
        self.dense1 = nn.Linear(64, 128)
        # Relu go here
        self.drop = nn.Dropout(p=0.2)
        # Step 5: Final Dense layer, softmax
        self.dense2 = nn.Linear(128, class_no)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.batch1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool2(out)
        out = self.batch2(out)
        out = self.pool3(out)
        out = self.dense1(out)
        out = self.drop(out)
        out = self.dense2(out)
        out = self.softmax(out)
        
        return out

# Generate torch model
torchy = TorchBrain()
print("\n", torchy, sep='')

# Print out all the weight shapes so we can get the transposition right
# torch weights, bias, running_mean, running_var corresponds to keras gamma, beta, moving mean, moving average
print("\n")
print(wi[0], "| Torch:", torchy.conv1.weight.shape)
print(wi[1], "| Torch:", torchy.conv1.bias.shape)
print(wi[2], "| Torch:", torchy.batch1.weight.shape)
print(wi[3], "| Torch:", torchy.batch1.bias.shape)
print(wi[4], "| Torch:", torchy.batch1.running_mean.shape)
print(wi[5], "| Torch:", torchy.batch1.running_var.shape)
print(wi[6], "| Torch:", torchy.conv2.weight.shape)
print(wi[7], "| Torch:", torchy.conv2.bias.shape)
print(wi[8], "| Torch:", torchy.batch2.weight.shape)
print(wi[9], "| Torch:", torchy.batch2.bias.shape)
print(wi[10], "| Torch:", torchy.batch2.running_mean.shape)
print(wi[11], "| Torch:", torchy.batch2.running_var.shape)
print(wi[12], "| Torch:", torchy.dense1.weight.shape)
print(wi[13], "| Torch:", torchy.dense1.bias.shape)
print(wi[14], "| Torch:", torchy.dense2.weight.shape)
print(wi[15], "| Torch:", torchy.dense2.bias.shape)

# Copy over all the weights
# Correct transposing is needed here.
torchy.conv1.weight.data = torch.from_numpy(np.transpose(weights[0]))
torchy.conv1.bias.data = torch.from_numpy(weights[1])
torchy.batch1.weight.data = torch.from_numpy(weights[2])
torchy.batch1.bias.data = torch.from_numpy(weights[3])
torchy.batch1.running_mean.data = torch.from_numpy(weights[4])
torchy.batch1.running_var.data = torch.from_numpy(weights[5])
torchy.conv2.weight.data = torch.from_numpy(np.transpose(weights[6]))
torchy.conv2.bias.data = torch.from_numpy(weights[7])
torchy.batch2.weight.data = torch.from_numpy(weights[8])
torchy.batch2.bias.data = torch.from_numpy(weights[9])
torchy.batch2.running_mean.data = torch.from_numpy(weights[10])
torchy.batch2.running_var.data = torch.from_numpy(weights[11])
torchy.dense1.weight.data = torch.from_numpy(np.transpose(weights[12]))
torchy.dense1.bias.data = torch.from_numpy(weights[13])
torchy.dense2.weight.data = torch.from_numpy(np.transpose(weights[14]))
torchy.dense2.bias.data = torch.from_numpy(weights[15])

print("\nAll done.")