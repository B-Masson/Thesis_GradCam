# Observe the data stored in the weights, as well as in the generated grad maps
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow import keras
from keras.models import load_model
import nibabel as nib
print("Imports working.")

display_mode = False

model_name = "ADModel_TWIN_v1.5_strip.h5"
model = load_model(model_name)
print("Keras model loaded in.")

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()
wi = [] # Weight info

if display_mode:
    for name, weight in zip(names, weights):
        string = name +" " +str(weight.shape)
        wi.append(string)
    for element in wi:
        print(element)

layer = "conv3d/kernel:0"
ind_1 = names.index(layer)
layer2 = "conv3d/bias:0"
ind_2 = names.index(layer2)
#print(layer, "- shape", weights[ind_1].shape, "-", weights[ind_1][0])
#print(layer2, "- shape", weights[ind_2].shape, "-", weights[0])

grads = nib.load("Grad-Maps\\conv\\attention_map_0_0_0.nii.gz").get_fdata()
print(grads.shape)
print(grads)
if grads.any() != 0:
    print("non-zero")
else:
    print("all zero")
