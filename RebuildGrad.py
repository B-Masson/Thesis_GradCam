# Can you solve all my problems, O Activation Wizard?
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print("Imports working.")

# Flags
display_mode = True # Print out all the weight info
single = True
memory_mode = False # Print out memory summaries
strip_mode = False
basic_mode = True
saving = True

# Memory setup
if memory_mode:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    GPUtil.showUtilization()

priority_slices = [56, 57, 58, 64, 75, 85, 88, 89, 96]
slice_range = np.arange(50, 100)
# Just pick a single model for now
#modelnum = 58
#model_name = "Models/2DSlice_V2-prio/model-"+str(modelnum)+".h5"
model_dir = "Models/2DSlice_V1.5-entire/model-"
#model = load_model(model_name)
#print("Keras model loaded in. [", model_name, "]")

#print("Compiling...")
#optim = keras.optimizers.Adam(learning_rate=0.001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later
#model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

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

# Incredibly dirty way of preparing this data
func = nib.load(path[0])
image_raw = np.asarray((func).get_fdata(dtype='float32'))
image_raw = ne.organiseADNI(image_raw, w, h, d, strip=strip_mode)
lab = to_categorical(labels[0])
print("Data obtained. Moving on to models...")

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
        #print("I HAVE OPTED TO NOT RESIZE SHIT (FOR NOW)")
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

# Vars
actloc = "Activations/2D/"
gradloc = "Gradients/2D/"
depth = 5
actvolume = np.zeros((167, 206, depth))
gradvolume = []
eps=1e-8

#for slicenum in slice_range:
for i in range(depth):
    slicenum = slice_range[i]
    model_name = model_dir + str(slicenum)
    model = load_model(model_name)
    print("Keras model loaded in. [", model_name, "]")
    #model.summary()
    #print("Compiling...")
    optim = keras.optimizers.Adam(learning_rate=0.001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    # Prep slice
    image = image_raw[:,:,slicenum]
    image = np.expand_dims(image, axis=0)
    #print("Image shape:", image.shape)
    # Here goes nothing
    if i == 0:
        layername = "conv2d"
    else:
        layername = "conv2d_"+str(i)
    print("Instantiating Grad Map...")
    icam = GradCAM(model, labels[0], layername)
    print("Generating heatmap.")
    grad = icam.compute_heatmap(image)
    print("Heatmap is shape:", grad.shape)
    # Save slice
    print("Displaying...")
    plt.imshow(grad)
    slicename = gradloc + "class" +str(labels[0]) +"_slice" +str(slicenum) +".png"
    plt.savefig(slicename)
    plt.clf()
    actvolume[:,:,i] = grad
    '''
    kerpred = model.predict(image)
    kerpreddy = np.argmax(kerpred, axis=1)
    print("Keras prediction:", kerpred[0], "| (", kerpreddy[0], "vs. actual:", labels[i], ")")
    kp.append(kerpreddy[0])
    '''

#np.swapaxes(actvolume, axis1, axis2)
print("Activation volume shape:", actvolume.shape)
new_image = nib.Nifti1Image(actvolume, func.affine)
nib.save(new_image, "Gradients/2D/AD-flat.nii.gz")
keras.backend.clear_session()
print("\nAll done.")


