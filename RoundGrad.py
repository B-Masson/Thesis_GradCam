import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import NIFTI_Engine as ne
import nibabel as nib
import matplotlib.pyplot as plt
import GPUtil
import sys
from collections import Counter
from tensorflow.keras.models import Model
print("Imports working.")

# Get da model
classy = "CN"

# Just pick a single model for now
model_name = "Models/ADModel_NEO_V5-basicvalid.h5"
model = load_model(model_name)
print("Keras model loaded in.")

# Get da input
class_param = ""
if classy == "CN":
    class_param = "-CN"
imgname = "Directories\\single"+class_param+".txt"
labname = "Directories\\single"+class_param+"-label.txt"
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
print("Distribution:", Counter(labels))

w = 169
h = 208
d = 179

# Prepare input
img = np.asarray(nib.load(path[0]).get_fdata(dtype='float32'))
img = ne.organiseADNI(img, w, h, d, strip=False)
img = np.expand_dims(img, axis=0)

# REF CODE START HERE
# --------------------------------
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        print("Setting up gradient model...")
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        print("Setting up Tape...")
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            
            loss = predictions[:, tf.argmax(predictions[0])]
    
        # use automatic differentiation to compute the gradients
        print("Computing gradients...")
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        print("Casting...")
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        print("Discarding batch dimension...")
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        print("Weight computations...")
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        print("I HAVE OPTED TO NOT RESIZE SHIT (FOR NOW)")
        (w, h) = (image.shape[2], image.shape[1])
        #print("Cam shape:", cam.numpy().shape)
        #heatmap = cv2.resize(cam.numpy(), (w, h))
        heatmap = cam.numpy()
        #print("Heatmap shape:", heatmap.shape)
        
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        print("Normalizing...")
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
    
    def heatmap_only(self, heatmap, colormap=cv2.COLORMAP_VIRIDIS):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        return heatmap
# --------------------------------
# REF CODE END HERE

# Predict and get grad
print("Image shape:", img.shape)
print("Class:", labels[0])
prediction = model.predict(img)
print("Prediction:", prediction[0])
i = np.argmax(prediction[0])

#for idx in range(len(model.layers)):
#    print(model.get_layer(index = idx).name)
print("Instantiating Grad Map...")
icam = GradCAM(model, i, 'conv3d')
#print("Generating heatmap.")
heatmap = icam.compute_heatmap(img)
print("Heatmap is shape:", heatmap.shape)
print("Img is:", img.shape)

#image = np.squeeze(img, axis=3)
#image = np.squeeze(image, axis=0)
#print("Img is:", image.shape)
#image = (image * 255).astype("uint8")
#image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
#print("Img is:", image.shape)

#(heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)
slicerange = range(25, 156, 25)
print("Time to try and display:")
for sliceno in slicerange:
    print("Slice", sliceno)
    heatmap2d = heatmap[:,:,sliceno]
    heatmap2d = icam.heatmap_only(heatmap2d, colormap=cv2.COLORMAP_INFERNO)
    plt.imshow(heatmap2d)
    plt.show()
    plt.clf()

'''
print("Displaying...")
fig, ax = plt.subplots(1, 3)
ax[0].imshow(heatmap)
ax[1].imshow(image)
ax[2].imshow(output)
plt.show()
fig.savefig("Grad_NEO/"+classy+"-map-slice-"+str(modelnum)+".png")
'''

print("All done!")