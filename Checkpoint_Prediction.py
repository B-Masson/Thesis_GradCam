# Testing class for running predictions on each checkpoint for a given model
print("EVALUATING CHECKPOINTS")
print("Importing packages...")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
from scipy import ndimage
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import NIFTI_Engine as ne
import nibabel as nib
from collections import Counter
import sys
print("Imports done.")

# Params
eval = True
batch_size = 2
norm_mode = False
strip_mode = False
w = 169
h = 208
d = 179
classNo = 2

# Fetch the saved testing data
testloc = "testing_saved.npz"
test = np.load(testloc)
x_test = test['a']
y_test = test['b']
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
print("Data loaded.")
y_hat = np.argmax(y_test)
print("Image count:", len(x_test))
#print("Testing distribution:", Counter(y_hat))

# Shuffle the test sets (for more varied testing)
x_test, y_test = shuffle(np.array(x_test), np.array(y_test))
print("Data shuffled.")

# Load up the model
def gen_basic_model(width=208, height=240, depth=256, classes=3): # Baby mode
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    x = layers.Conv3D(filters=32, kernel_size=5, padding='same', activation="relu")(inputs) # Layer 1: Simple 32 node start
    #x = layers.Conv3D(filters=32, kernel_size=5, padding='same', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation="relu")(inputs) # Layer 1: Simple 32 node start
    #x = layers.Conv3D(filters=32, kernel_size=5, padding='valid', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation="relu")(inputs) # Layer 1: Simple 32 node star
    x = layers.MaxPool3D(pool_size=5, strides=5)(x) # Usually max pool after the conv layer
    
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units

    outputs = layers.Dense(units=classNo, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN_Basic")

    return model

def gen_spicy_model(width=169, height=208, depth=179, classes=2):
    modelname = "Spicy-CNN"
    print(modelname)
    inputs = keras.Input((width, height, depth, 1))
    
    x = layers.Conv3D(filters=8, kernel_size=5, padding='valid', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv3D(filters=16, kernel_size=5, padding='valid', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv3D(filters=32, kernel_size=5, padding='valid', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv3D(filters=64, kernel_size=5, padding='valid', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    
    outputs = layers.Dense(units=classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name=modelname)
    
    return model

model = gen_spicy_model(w, h, d, classes=2)
optim = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

def load_val(file, label): # NO AUG
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    if norm_mode:
        nifti = ne.resizeADNI(nifti, w, h, d, stripped=True)
    else:
        nifti = ne.organiseADNI(nifti, w, h, d, strip=strip_mode)
    nifti = tf.convert_to_tensor(nifti, np.float32)

    return nifti, label

def load_test(file): # NO AUG, NO LABEL
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    if norm_mode:
        nifti = ne.resizeADNI(nifti, w, h, d, stripped=True)
    else:
        nifti = ne.organiseADNI(nifti, w, h, d, strip=strip_mode)
    nifti = tf.convert_to_tensor(nifti, np.float32)
    return nifti

def load_val_wrapper(file, labels):
    return tf.py_function(load_val, [file, labels], [np.float32, np.float32])

def load_test_wrapper(file):
    return tf.py_function(load_test, [file], [np.float32])

test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_x = tf.data.Dataset.from_tensor_slices((x_test))

try:
    test_set = (
        test.map(load_val_wrapper)
        .batch(batch_size)
        .prefetch(batch_size)
    ) # Later we may need to use a different wrapper function? Not sure.
except:
    print("Couldn't assign test_set to a wrapper (for evaluation).")

try:
    test_set_x = (
        test_x.map(load_test_wrapper)
        .batch(batch_size)
        .prefetch(batch_size)
    )
    #if not testing_mode: # NEED TO REWORK THIS
except Exception as e:
    print("Couldn't assign test_set_x to a wrapper (for matrix). Reason:", e)

'''
import glob
checkpoint_dir = "/scratch/mssric004/Checkpoints/"
checkpoint_pattern = checkpoint_dir+"basic*.ckpt*"
checkpoints = glob.glob(checkpoint_pattern)
print("Checkpoints:", checkpoints)
'''
root = "/scratch/mssric004/Checkpoints/"
mutor = "kfold-spice-"
dir = os.listdir(root)
checks = 0
for file in dir:
    if (".index" in file) and (mutor in file):
        checks += 1
root = root + mutor

if eval:
    count = 1
    print("\nConducting the evaluation:\n--------------------")
    while count <= checks:
        if count < 10:
            num = "0" + str(count)
        else:
            num = str(count)
        checkpoint = root + num +".ckpt"
        #print("Checkpoint:", checkpoint)
        model.load_weights(checkpoint)
        scores = model.evaluate(test_set, verbose=0) # Should not need to specify batch size, because of set
        acc = scores[1]*100
        loss = scores[0]
        print("Checkpoint", count, "scores - Acc:", acc, "Loss:", loss)
        count += 1
else:
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
    count = int(input("What checkpoint do we want to access?\n"))
    if count < 10:
        checknum = "0" + str(count)
    else:
        checknum = str(count)
    checkpoint = root + checknum +".ckpt"
    model.load_weights(checkpoint)
    scores = model.evaluate(test_set, verbose=0) # Should not need to specify batch size, because of set
    acc = scores[1]*100
    loss = scores[0]
    print("Checkpoint", checknum, "scores - Acc:", acc, "Loss:", loss)
    print("\nGenerating classification report...")
    try:
        y_pred = model.predict(test_set_x, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        rep = classification_report(y_test, y_pred)
        conf = confusion_matrix(y_test, y_pred)
        coh = cohen_kappa_score(y_test, y_pred)
        print(rep)
        print("\nConfusion matrix:")
        print(conf)
        print("Cohen Kappa Score (0 = chance, 1 = perfect):", coh)
        limit = min(30, len(y_test))
        print("\nActual test set (first ", (limit+1), "):", sep='')
        print(y_test[:limit])
        print("Predictions are  as follows (first ", (limit+1), "):", sep='')
        print(y_pred[:limit])
    except:
        print("Error occured in classification report (ie. predict). Test set labels are:\n", y_test)

print("Done.")
