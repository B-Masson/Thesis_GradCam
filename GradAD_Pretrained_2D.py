# Code to generate GradCam maps for a given 2D brain slice, using a pretrained 2D model
# Richard Masson
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, DataLoader
import torch
from torchsummary import summary
import cv2
from medcam import medcam
import NIFTI_Engine as ne
import numpy as np
import matplotlib.pyplot as plt
print("Imports working.")

# Setup the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet152(pretrained=True)
model.to(device=device)
model.eval()

# Read in the data slices
w = 208
h = 240
d = 256
print("Extracting data")
x, y = ne.extractSlices(w, h, d, "Data\ADNI_Test", 3, exclusions="AD", limit=1)
#plt.imshow(x[0], cmap='bone') # Can't use because we have channels first rn
#plt.show() # Later change to save the image instead

# Convert into a data loader
#x = np.array(x)
#y = np.array(y)
x = np.array(x)
tensor_x = torch.Tensor(x) # transform to torch tensor
tensor_x = tensor_x.to(device=device)
#tensor_y = torch.Tensor(y)
data = TensorDataset(tensor_x) # create your datset

data_loader = DataLoader(data, batch_size= 1, shuffle=False) # Set shuffle to true when we want to look at more than just the first instance.

# Cam injection
print("Done. Attempting injection...")
model = medcam.inject(model, output_dir='Grad-Maps-2D', backend='gcam', layer='layer4', label='best', save_maps=True)
print("Injection successful.")

for i in range (len(data_loader)):
    batch = next(iter(data_loader))
    _ = model(batch[0])

print("All done.")
