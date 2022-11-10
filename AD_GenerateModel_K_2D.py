# Messing around with stuff without breaking the original version of the code.
# Richard Masson
# Info: Trying to fix the model since I'm convinced it's scuffed.
# Last use in 2021: October 29th
print("\nIMPLEMENTATION: 2D K-Fold")
desc = "Advanced model. Let's go."
print(desc)
import os
import subprocess as sp
from time import perf_counter # Memory shit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import nibabel as nib
import NIFTI_Engine as ne
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
#print("TF Version:", tf.version.VERSION)
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import sys
import random
import datetime
from collections import Counter
from volumentations import *
from sklearn.model_selection import KFold, StratifiedKFold
import glob
print("Imports working.")

# Attempt to better allocate memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
'''
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
'''
from datetime import date
print("Today's date:", date.today())

# Are we in testing mode?
testing_mode = False
memory_mode = False
limiter = False
pure_mode = False
strip_mode = False
norm_mode = False
curated = False
trimming = True
bad_data = False
nosplit = True
logname = "2DK_V6-advanced-TEST"
modelname = "ADModel_"+logname
if not testing_mode:
    print("MODELNAME:", modelname)
    print("LOGS CAN BE FOUND UNDER", logname)

# Model hyperparameters
if testing_mode:
    epochs = 3 #Small for testing purposes
    batch_size = 3
else:
    epochs = 2 # JUST FOR NOW
    batch_size = 3 # Going to need to fiddle with this over time (balance time save vs. running out of memory)

# Define image size (lower image resolution in order to speed up for broad testing)
if testing_mode:
    scale = 2
else:
    scale = 1 # For now
w = int(169/scale) # 208 # 169
h = int(208/scale) # 240 # 208
d = int(179/scale) # 256 # 179
tic = perf_counter()

# Prepare parameters for fetching the data
modo = 2 # 1 for CN/MCI, 2 for CN/AD, 3 for CN/MCI/AD, 4 for weird AD-only, 5 for MCI-only
if modo == 3 or modo == 4:
    #print("Setting for 3 classes")
    classNo = 3 # Expected value
else:
    #print("Setting for 2 classes")
    classNo = 2 # Expected value
if testing_mode: # CHANGIN THINGS UP
	filename = ("Directories/test_adni_" + str(modo)) # CURRENTLY AIMING AT TINY ZONE
elif pure_mode:
    filename = ("Directories/test_tiny_adni_" + str(modo)) # CURRENTLY AIMING AT TINY ZONE
else:
    filename = ("Directories/adni_" + str(modo))
if testing_mode:
    print("TEST MODE ENABLED.")
elif limiter:
    print("LIMITERS ENGAGED.")
if curated:
    print("USING CURATED DATA.")
if pure_mode:
    print("PURE MODE ENABLED.")
if trimming:
    print("TRIMMING DOWN CLASSES TO PREVENT IMBALANCE")
if norm_mode:
    print("USING NORMALIZED, STRIPPED IMAGES.")
elif strip_mode:
    print("USING STRIPPED IMAGES.")
if bad_data:
    filename = "Directories/baddata_adni_" + str(modo)
print("Filepath is", filename)
if curated:
    imgname = "Directories/curated_images.txt"
    labname = "Directories/curated_labels.txt"
elif norm_mode:
    imgname = filename+"_images_normed.txt"
    labname = filename+"_labels_normed.txt"
elif strip_mode:
    if trimming:
        imgname = filename+"_trimmed_images_stripped.txt"
        labname = filename+"_trimmed_labels_stripped.txt"
    else:
        imgname = filename+"_images_stripped.txt"
        labname = filename+"_labels_stripped.txt"
elif trimming:
    imgname = filename+"_trimmed_images.txt"
    labname = filename+"_trimmed_labels.txt"
else:
    imgname = filename + "_images.txt"
    labname = filename + "_labels.txt"

# Grab the data
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
print("Data distribution:", Counter(labels))
labels = to_categorical(labels, num_classes=classNo, dtype='float32')
# ^ for k, don't do it here
print("\nOBTAINED DATA. (Scaling by a factor of ", scale, ")", sep='')

# Split data
'''
rar = 1
if testing_mode:
    x, x_test, y, y_test = train_test_split(path, labels, test_size=0.25, stratify=labels, random_state=rar, shuffle=True) # 75/25 (for eventual 50/25/25)
else:
    x, x_test, y, y_test = train_test_split(path, labels, test_size=0.2, stratify=labels, random_state=rar, shuffle=True) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
x = np.array(x)
y = np.array(y)
'''
# Trying out something weird
rar = 0
if testing_mode:
    x_train, x_val, y_train, y_val = train_test_split(path, labels, test_size=0.5, stratify=labels, random_state=rar, shuffle=True) # 50/50 (for eventual 50/25/25)
else:
    x_train, x_val, y_train, y_val = train_test_split(path, labels, stratify=labels, random_state=rar, shuffle=True) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
if testing_mode:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.5, random_state=rar, shuffle=True) # Just split 50/50.
else:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, random_state=rar, test_size=0.4, shuffle=True) # 60/40 val/test
# Now stitch them together like Frankenstein
x = x_train + x_val
traintemp = np.argmax(y_train, axis=1).tolist()
valtemp = np.argmax(y_val, axis=1).tolist()
y = traintemp+valtemp
x = np.array(x)
y = np.array(y)

# To observe data distribution
def countClasses(categors, name):
    temp = np.argmax(categors, axis=1)
    print(name, "distribution:", Counter(temp))

print("Number of training/validation images:", len(x))
#countClasses(y, "Training/validation")
print("Number of testing images:", len(x_test), "\n")
if testing_mode:
    print("Training labels:", y)
#print("Label type:", y[0].dtype)

# Data augmentation functions
aug_rate = 1
def get_augmentation(patch_size):
    return Compose([
        Rotate((-3, 3), (-3, 3), (-3, 3), p=0.6), #0.5
        #Flip(2, p=1)
        ElasticTransform((0, 0.05), interpolation=2, p=0.3), #0.1
        #GaussianNoise(var_limit=(1, 1), p=1), #0.1
        RandomGamma(gamma_limit=(0.6, 1), p=0) #0.4
    ], p=aug_rate) #0.9 #NOTE: Temp not doing augmentation. Want to take time to observe the effects of this stuff
aug = get_augmentation((w,h,d)) # For augmentations

def load_image(file, label):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    if norm_mode:
        nifti = ne.resizeADNI(nifti, w, h, d, stripped=True)
    else:
        nifti = ne.organiseADNI(nifti, w, h, d, strip=strip_mode)

    # Augmentation
    data = {'image': nifti}
    aug_data = aug(**data)
    nifti = aug_data['image']

    nifti = tf.convert_to_tensor(nifti, np.float32)

    if nifti.shape == (5,23,8,1):
        print("Why the hell is the shape", nifti.shape, "??")
        print("File in question:", loc)
        sys.exit()
    return nifti, label

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

def load_image_wrapper(file, labels):
    return tf.py_function(load_image, [file, labels], [np.float32, np.float32])

def load_val_wrapper(file, labels):
    return tf.py_function(load_val, [file, labels], [np.float32, np.float32])

def load_test_wrapper(file):
    return tf.py_function(load_test, [file], [np.float32])

def fix_shape(images, labels):
    images.set_shape([None, w, h, d, 1])
    labels.set_shape([images.shape[0], classNo])
    return images, labels

def fix_dims(image):
    image.set_shape([None, w, h, d, 1])
    return image

def fix_wrapper(file, labels):
    return tf.py_function(fix_shape, [file, labels], [np.float32, np.float32])

print("Quickly preparing test data...")

# Prepare the test data over here first
test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_x = tf.data.Dataset.from_tensor_slices((x_test))

test_set_x = (
    test_x.map(load_test_wrapper)
    .batch(batch_size)
    .map(fix_dims)
    .prefetch(batch_size)
)
test_set = (
    test.map(load_val_wrapper)
    .batch(batch_size)
    .map(fix_shape)
    .prefetch(batch_size)
) # Later we may need to use a different wrapper function? Not sure.

print("Test data prepared.")

# Model architecture go here
# For consideration: https://www.frontiersin.org/articles/10.3389/fbioe.2020.534592/full#B22
# Current inspiration: https://ieeexplore.ieee.org/document/7780459 (VGG19)
def gen_model(width=208, height=240, depth=256, classes=3): # Make sure defaults are equal to image resizing defaults
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    x = layers.Conv3D(filters=8, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x) # Paper conv and BN go together, then pooling
    #x = layers.Dropout(0.1)(x) # Apparently there's merit to very light dropout after each conv layer
    
    # kernel_regularizer=l2(0.01)
    x = layers.Conv3D(filters=16, kernel_size=3, strides=1, padding="same", activation="relu")(x) #(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    #x = layers.Dropout(0.1)(x)
    # NOTE: RECOMMENTED LOL

    x = layers.Conv3D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    #x = layers.Dropout(0.1)(x)
    # NOTE: Also commented this one for - we MINIMAL rn
    
    x = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    #x = layers.Dropout(0.1)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    #x = layers.Dropout(0.1)(x)

    x = layers.Dropout(0.5)(x)
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    #x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units
    x = layers.Dense(units=1300)(x)
    x = layers.Dense(units=50)(x)
    #x = layers.Dropout(0.3)(x) # Start low, and work up if overfitting seems to be present

    outputs = layers.Dense(units=classNo, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")


    return model

def gen_model_2(width=208, height=240, depth=256, classes=3): # Make sure defaults are equal to image resizing defaults
    print("USING ADVANCED MODEL")
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    x = layers.Conv3D(filters=8, kernel_size=3, padding="same", activation="relu")(inputs)
    #x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x) # Paper conv and BN go together, then pooling
    x = layers.Dropout(0.1)(x) # Apparently there's merit to very light dropout after each conv layer
    
    # kernel_regularizer=l2(0.01)
    x = layers.Conv3D(filters=16, kernel_size=3, padding="same", activation="relu")(x) #(x)
    #x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.1)(x)
    # NOTE: RECOMMENTED LOL

    x = layers.Conv3D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    #x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.1)(x)
    # NOTE: Also commented this one for - we MINIMAL rn
    
    x = layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    #x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.1)(x)

    #x = layers.Conv3D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    #x = layers.Dropout(0.1)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    #x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units
    x = layers.Dense(units=128)(x)
    x = layers.Dense(units=64)(x)
    #x = layers.Dropout(0.3)(x) # Start low, and work up if overfitting seems to be present

    outputs = layers.Dense(units=classNo, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")


    return model

def gen_basic_model(width=208, height=240, depth=256, classes=3): # Baby mode
    print("USING BASIC MODEL")
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    x = layers.Conv3D(filters=32, kernel_size=5, padding='same', activation="relu")(inputs) # Layer 1: Simple 32 node start
    #x = layers.Conv3D(filters=32, kernel_size=5, padding='valid', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation="relu")(inputs) # Layer 1: Simple 32 node start
    x = layers.MaxPool3D(pool_size=5, strides=5)(x) # Usually max pool after the conv layer
    
    #x = layers.Dropout(0.5)(x) # Here or below?
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units

    outputs = layers.Dense(units=classNo, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN_Basic")

    return model

def gen_basic_noreg_model(width=208, height=240, depth=256, classes=3): # Baby mode
    print("USING BASIC MODEL (NO REG)")
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    #x = layers.Conv3D(filters=32, kernel_size=5, padding='same', activation="relu")(inputs) # Layer 1: Simple 32 node start
    x = layers.Conv3D(filters=32, kernel_size=5, padding='same', activation="relu")(inputs) # Layer 1: Simple 32 node start
    x = layers.MaxPool3D(pool_size=5, strides=5)(x) # Usually max pool after the conv layer
    
    #x = layers.Dropout(0.5)(x) # Here or below?
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units

    outputs = layers.Dense(units=classNo, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN_Basic_NoReg")

    return model

def gen_advanced_sep_model(width=169, height=208, depth=179, classes=2):
    modelname = "Advanced-2D-Separable-CNN"
    print(modelname)
    inputs = keras.Input((width, height, depth))
    
    x = layers.SeparableConv2D(filters=8, kernel_size=5, padding='valid', activation='relu', data_format="channels_last")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.SeparableConv2D(filters=16, kernel_size=5, padding='valid', activation='relu', data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.SeparableConv2D(filters=32, kernel_size=5, padding='valid', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu', data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.SeparableConv2D(filters=64, kernel_size=5, padding='valid', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu', data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    
    outputs = layers.Dense(units=classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name=modelname)
    
    return model

# Metrics
if batch_size > 1:
    metric = 'binary_accuracy'
else:
    metric = 'accuracy'

# Checkpointing & Early Stopping
mon = 'val_' +metric
es = EarlyStopping(monitor=mon, patience=10, restore_best_weights=True) # Temporarily turning this off because I want to observe the full scope
checkpointname = "/scratch/mssric004/Checkpoints/kfold-advanced-{epoch:02d}.ckpt"
if testing_mode:
    print("Setting checkpoint")
    checkpointname = "/TestCheckpoints/neo_checkpoint-{epoch:02d}.ckpt"
mc = ModelCheckpoint(checkpointname, monitor=mon, mode='auto', verbose=2, save_weights_only=True, save_best_only=False) #Maybe change to true so we can more easily access the "best" epoch
if testing_mode:
    log_dir = "/scratch/mssric004/test_logs/fit/neo/" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M")
else:
    if logname != "na":
        log_dir = "/scratch/mssric004/logs/fit/neo/" + logname + "_" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M")
    else:
        log_dir = "/scratch/mssric004/logs/fit/neo/" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M")#.strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Custom callbacks (aka make keras actually report stuff during training)
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        #keys = list(logs.keys())
        #print("End of training epoch {} of training; got log keys: {}".format(epoch, keys))
        print("Epoch {}/{} > ".format(epoch+1, epochs))
        #if (epoch+1) == epochs:
        #    print('')

class DebugCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        #keys = list(logs.keys())
        #print("End of training epoch {} of training; got log keys: {}".format(epoch, keys))
        print("Epoch {}/{} > ".format(epoch+1, epochs))
        #if (epoch+1) == epochs:
        #    print('')
    def on_train_batch_begin(self, batch, logs=None):
        print("...Training: start of batch {}".format(batch))
'''
# Setting class weights
from sklearn.utils import class_weight

y_org = y
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_org), y=y_org)
class_weight_dict = dict()
for index,value in enumerate(class_weights):
    class_weight_dict[index] = value
#class_weight_dict = {i:w for i,w in enumerate(class_weights)}
print("Class weight distribution will be:", class_weight_dict)
'''
# Build model. (Have to do all this here to try fix OOM issue)
def initial_model(w, h, d, classNo, metric):
    print("USING ADVANCED MODEL.")
    model = gen_advanced_sep_model(width=w, height=h, depth=d, classes=classNo)
    optim = keras.optimizers.Adam(learning_rate=0.0001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later
    if metric == 'binary_accuracy':
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()]) #metrics=['accuracy']) #metrics=[tf.keras.metrics.BinaryAccuracy()]
    else:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy']) #metrics=[tf.keras.metrics.BinaryAccuracy()]
    return model, model.get_weights()

model, initials = initial_model(w, h, d, classNo, metric)

def reset_weights(reused_model, init_weights):
    reused_model.set_weights(init_weights)

# K-Fold setup
n_folds = 5 # FOR NOW
if testing_mode:
    n_folds = 5
acc_per_fold = []
loss_per_fold = []
rar = 0
skf = StratifiedKFold(n_splits=n_folds, random_state=rar, shuffle=True)
mis_classes = []
suc_classes = []

fold = 0
# Start training
print("\nStarting cross-fold validation process...")
print("Params:", epochs, "epochs &", batch_size, "batches.")
for train_index, val_index in skf.split(x, y):
    fold = fold + 1
    print("***************\nNow on Fold", fold, "out of", n_folds)

    # Set up train-val split per fold
    x_train = x[train_index]
    x_val = x[val_index]
    y_train = y[train_index]
    y_val = y[val_index]

    # Have to convert the labels to categorical here since KFolds doesn't like that
    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)
    
    print("Training iteration on " + str(len(x_train)) + " training samples, " + str(len(x_val)) + " validation samples")

    # Give each fold a different local checkpoint
    localcheck = "/scratch/mssric004/TrueChecks/" + modelname +"_fold" +str(fold) +".ckpt"
    be = ModelCheckpoint(localcheck, monitor=mon, mode='auto', verbose=2, save_weights_only=True, save_best_only=True)

    print("Setting up dataloaders...")
    # TO-DO: Augmentation stuff
    # Data loaders
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_set = (
        train.shuffle(len(train))
        .map(load_image_wrapper)
        .batch(batch_size)
        .map(fix_shape)
        .prefetch(batch_size)
    )

    # Only rescale.
    validation_set = (
        val.shuffle(len(x_val))
        .map(load_val_wrapper)
        .batch(batch_size)
        .map(fix_shape)
        .prefetch(batch_size)
    )

    # Run the model
    if fold == 1:
        model.summary()

    # Reset the weights
    print("Resetting weights...")
    reset_weights(model, initials)

    print("Fitting model...")
    if testing_mode:
    #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0)
        history = model.fit(train_set, validation_data=validation_set, epochs=epochs, verbose=0, callbacks=[be, CustomCallback()]) # DON'T SPECIFY BATCH SIZE, CAUSE INPUT IS ALREADY A BATCHED DATASET
    else:
        #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0, shuffle=True)
        history = model.fit(train_set, validation_data=validation_set, epochs=epochs, callbacks=[es, be, CustomCallback()], verbose=0, shuffle=True)
        # Not saving checkpoints FOR NOW
    print("RESULTS FOR FOLD", fold, ":")
    print(history.history)
    
    # Readings
    try:
        print("\nAccuracy max:", round(max(history.history[metric])*100,2), "% (epoch", history.history[metric].index(max(history.history[metric]))+1, ")")
        print("Loss min:", round(min(history.history['loss']),2), "(epoch", history.history['loss'].index(min(history.history['loss']))+1, ")")
        print("Validation accuracy max:", round(max(history.history['val_'+metric])*100,2), "% (epoch", history.history['val_'+metric].index(max(history.history['val_'+metric]))+1, ")")
        print("Val loss min:", round(min(history.history['val_loss']),2), "(epoch", history.history['val_loss'].index(min(history.history['val_loss']))+1, ")")
    except Exception as e:
        print("Cannot print out summary data. Reason:", e)
    
    def make_unique(path, run):
        print("Figuring out where to save plots to...")
        expand = 1
        folder = path+"/run"+str(expand)+"/"
        while True:
            if os.path.isdir(folder):
                #print("I have determined that there is already a folder named", folder)
                expand += 1
                folder = path+"run"+str(expand)+"/"
                continue
            else:
                if run == 1:
                    print(folder, "does not exist, so we shall create it and save stuff there.")
                    os.mkdir(folder)
                else:
                    #print("Subsequent run, therefore just save to run", str(expand-1))
                    return path+"/run"+str(expand-1)+"/"
                break
        return folder

    plotting = not testing_mode
    if plotting:
        try:
            print("Importing matplotlib.")
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
            plotpath = "Plots/Kfold/"
            path = make_unique(plotpath, fold)
            if testing_mode:
                plotname = "testing_fold"
            else:
                plotname = "fold"
            plotname = plotname + "_" + str(fold)
            # Plot stuff
            plt.plot(history.history[metric])
            plt.plot(history.history[('val_'+metric)])
            plt.legend(['train', 'val'], loc='upper left')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            name = plotname + "_acc.png"
            plt.savefig(path+name)
            plt.clf()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.legend(['train', 'val'], loc='upper left')
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            name = plotname + "_loss.png"
            plt.savefig(path+name)
            plt.clf()
            #plt.savefig(plotname + "_val" + ".png")
            print("Saved plots to", path)
        except Exception as e:
            print("Plotting didn't work out. Error:", e)
    
    # Load best checkpoint
    model.load_weights(localcheck)
    
    # Final evaluation
    print("\nEvaluating using test data...")
    scores = model.evaluate(test_set, verbose=0) # Should not need to specify batch size, because of set
    acc = scores[1]*100
    loss = scores[0]
    print("Fold", fold, "evaluated scores - Acc:", acc, "Loss:", loss)
    acc_per_fold.append(acc)
    loss_per_fold.append(loss)

    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

    print("\nGenerating classification report...")
    y_pred = model.predict(test_set_x, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)
    y_test_arged = np.argmax(y_test, axis=1)
    rep = classification_report(y_test_arged, y_pred)
    print(rep)
    try:
        conf = confusion_matrix(y_test_arged, y_pred)
        coh = cohen_kappa_score(y_test_arged, y_pred)
        print("\nConfusion matrix:")
        print(conf)
        print("Cohen Kappa Score (0 = chance, 1 = perfect):", coh)
    except Exception as e:
        print("Something wrong with confusion matrix/CK score. Errror:", e)
    limit = min(30, len(y_test_arged))
    print("\nActual test set (first ", (limit+1), "):", sep='')
    print(y_test_arged[:limit])
    print("Predictions are  as follows (first ", (limit+1), "):", sep='')
    print(y_pred[:limit])

    # Add to the list of images bringing up issues
    try:
        for i in range(0, len(y_pred)):
            if y_pred[i] != y_test_arged[i]:
                mis_classes.append(x_test[i])
            else:
                suc_classes.append(x_test[i])
    except Exception as e:
        print("Error while checking incorrect predictions.", e)
        
    # Clean up checkpoints
    print("Cleaning up...")
    found = glob.glob(localcheck+"*")
    if len(found) == 0:
        print("The system cannot find", localcheck)
    else:
        removecount = 0
        for checkfile in found:
            removecount += 1
            os.remove(checkfile)
        print("Successfully cleaned up", removecount, "checkpoint files.")
        
    print("Average so far:" , np.mean(acc_per_fold), "+-", np.std(acc_per_fold))
    if nosplit:
        print("ENDING AFTER ONE FOLD.")
        break

# Save outside the loop my sire
if testing_mode:
    modelname = "ADModel_2DK_Testing"
else:
    modelname = modelname +".h5"
model.save("/scratch/mssric004/Saved Models/"+modelname)
print("Saved the model to scratch models:", modelname)
# Electing not to save for now since the file it generates is HUGE


# Average scores
print("------------------------------------------------------------------------")
print("Score per fold")
for i in range(0, len(acc_per_fold)):
  print("------------------------------------------------------------------------")
  print("Fold", i+1, "- Loss:", loss_per_fold[i], "- Accuracy:", acc_per_fold[i])
print("------------------------------------------------------------------------")
print("Average scores for all folds:")
print("Accuracy:", np.mean(acc_per_fold), "+-", np.std(acc_per_fold))
print("Loss:",  np.mean(loss_per_fold), "+-", np.std(loss_per_fold))
print("------------------------------------------------------------------------")

# Save stuff so I can make a box and whisker plot later
if testing_mode:
    loc = "Means/KTesting"
else:
    loc = "Means/" + logname
print("Saving means to:", loc)
np.savez(loc, acc_per_fold, loss_per_fold)

# Check those tricky images
mis_d = Counter(mis_classes)
howmany = 3
print("\nMismatched test images (top ", howmany, "):", sep='')
common = mis_d.most_common()
for j in range(0, howmany-1):
    print(common[j])

# Best performing images
suc_d = Counter(suc_classes)
print("\nBest performing images (top ", howmany, ":", sep='')
least = mis_d.most_common()
for k in range(0, howmany-1):
    print(least[-k])

toc = perf_counter()
total_seconds = (int) (toc-tic)
total_time = datetime.timedelta(seconds=total_seconds)

print("Done. (Total time:", total_time, ")")