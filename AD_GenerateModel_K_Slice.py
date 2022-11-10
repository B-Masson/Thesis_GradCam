# Messing around with stuff without breaking the original version of the code.
# Richard Masson
# Info: Trying to fix the model since I'm convinced it's scuffed.
# Last use in 2021: October 29th
print("\nIMPLEMENTATION: K-Fold 2D Slices")
desc = "First run of kfold slices. Basic model."
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
from sklearn.metrics import accuracy_score
from statistics import mode, mean
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
testing_mode = True
memory_mode = False
limiter = False
pure_mode = False
strip_mode = False
norm_mode = False
curated = False
trimming = True
bad_data = False
#modelname = "ADModel_2DK_v1-bad-data" #Next in line: ADMODEL_NEO_v1.3
logname = "SliceK_V3-trial" #Neo_V1.3
modelname = "ADModel_"+logname
if not testing_mode:
    print("MODELNAME:", modelname)
    print("LOGS CAN BE FOUND UNDER", logname)

# Model hyperparameters
if testing_mode:
    epochs = 1 #Small for testing purposes
    batch_size = 1
else:
    epochs = 30 # JUST FOR NOW
    batch_size = 1 # Going to need to fiddle with this over time (balance time save vs. running out of memory)

# Set which slices to use, based on previous findings
priority_slices = [56, 57, 58, 64, 75, 85, 88, 89, 96]

# Define image size (lower image resolution in order to speed up for broad testing)
if testing_mode:
    scale = 2
else:
    scale = 1 # For now
w = int(169/scale) # 208 # 169
h = int(208/scale) # 240 # 208
d = int(179/scale) # 256 # 179

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
#labels = to_categorical(labels, num_classes=classNo, dtype='float32')
# ^ for k, don't do it here
print("\nOBTAINED DATA. (Scaling by a factor of ", scale, ")", sep='')

# Split data
rar = 0
if testing_mode:
    x, x_test, y, y_test = train_test_split(path, labels, test_size=0.25, stratify=labels, random_state=rar, shuffle=True) # 75/25 (for eventual 50/25/25)
else:
    x, x_test, y, y_test = train_test_split(path, labels, test_size=0.1, stratify=labels, random_state=rar, shuffle=True) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
x = np.array(x)
y = np.array(y)

# Need to make sure y_test is already prepared
y_test = to_categorical(y_test, num_classes=classNo, dtype='float32')

# To observe data distribution
def countClasses(categors, name):
    #temp = np.argmax(categors, axis=1)
    print(name, "distribution:", Counter(categors))

print("Number of training/validation images:", len(x))
countClasses(y, "Training/validation")
#y_train = np.asarray(y_train)
#print("Validation distribution:", Counter(y_val))
print("Number of testing images:", len(x_test), "\n")

# Data augmentation functions
def get_augmentation(patch_size):
    return Compose([
        Rotate((-3, 3), (-3, 3), (-3, 3), p=0.6), #0.5
        #Flip(2, p=1)
        ElasticTransform((0, 0.05), interpolation=2, p=0.3), #0.1
        #GaussianNoise(var_limit=(1, 1), p=1), #0.1
        RandomGamma(gamma_limit=(0.6, 1), p=0) #0.4
    ], p=1) #0.9 #NOTE: Temp not doing augmentation. Want to take time to observe the effects of this stuff
aug = get_augmentation((w,h,d)) # For augmentations

def load_image(file, label):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    if norm_mode:
        nifti = ne.resizeADNI(nifti, w, h, d, stripped=True)
    else:
        nifti = ne.organiseADNI(nifti, w, h, d, strip=strip_mode)
    data = {'image': nifti}
    aug_data = aug(**data)
    nifti = aug_data['image']
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

def load_slice(file, label):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    #print("using slice", n)
    slice = nifti[:,:,n]
    slice = ne.organiseSlice(slice, w, h, strip=strip_mode)
    # Augmentation
    # TO DO
    slice = tf.convert_to_tensor(slice, np.float32)
    return slice, label

def load_testslice(file):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    #print("using slice", n)
    slice = nifti[:,:,n]
    slice = ne.organiseSlice(slice, w, h, strip=strip_mode)
    # Augmentation
    # TO DO
    slice = tf.convert_to_tensor(slice, np.float32)
    return slice

def load_image_wrapper(file, labels):
    return tf.py_function(load_image, [file, labels], [np.float32, np.float32])

def load_test_wrapper(file):
    return tf.py_function(load_test, [file], [np.float32])

def load_slice_wrapper(file, labels):
    return tf.py_function(load_slice, [file, labels], [np.float32, np.float32])

def load_testslice_wrapper(file):
    return tf.py_function(load_testslice, [file], [np.float32])

# This needs to exist in order to allow for us to use an accuracy metric without getting weird errors
def fix_shape(images, labels):
    images.set_shape([None, w, h, 1])
    labels.set_shape([1, classNo])
    return images, labels

def fix_dims(image):
    image.set_shape([None, w, h, d, 1])
    return image

print("Quickly preparing test data...")

# Prepare the test data over here first
test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_x = tf.data.Dataset.from_tensor_slices((x_test))

test_set_x = (
        test_x.map(load_testslice_wrapper)
        .batch(batch_size)
        #.map(fix_dims)
        .prefetch(batch_size)
)
test_set = (
    test.map(load_slice_wrapper)
    .batch(batch_size)
    #.map(fix_shape)
    .prefetch(batch_size)
)

print("Test data prepared.")

# Model architecture go here
# For consideration: https://www.frontiersin.org/articles/10.3389/fbioe.2020.534592/full#B22
# Current inspiration: https://ieeexplore.ieee.org/document/7780459 (VGG19)
def gen_basic_2Dmodel(width=208, height=240, depth=256, classes=3): # Baby mode
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth)) # Added extra dimension in preprocessing to accomodate that 4th dim

    #x = layers.Conv3D(filters=32, kernel_size=5, padding='same', activation="relu")(inputs) # Layer 1: Simple 32 node start
    x = layers.SeparableConv2D(filters=32, kernel_size=5, padding='valid', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation="relu", data_format="channels_last")(inputs) # Layer 1: Simple 32 node start
    x = layers.MaxPool2D(pool_size=5, strides=5)(x) # Usually max pool after the conv layer
    
    #x = layers.Dropout(0.5)(x) # Here or below?
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units

    outputs = layers.Dense(units=classNo, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="2DCNN_Basic")

    return model

def gen_advanced_2d_model(width=169, height=208, depth=179, classes=2):
    modelname = "Advanced-2D-CNN"
    #print(modelname)
    inputs = keras.Input((width, height, depth))
    
    x = layers.Conv2D(filters=8, kernel_size=5, padding='valid', activation='relu', data_format="channels_last")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(filters=16, kernel_size=5, padding='valid', activation='relu', data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(filters=32, kernel_size=5, padding='valid', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu', data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(filters=64, kernel_size=5, padding='valid', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu', data_format="channels_last")(x)
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

# Custom callbacks (aka make keras actually report stuff during training)
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        #keys = list(logs.keys())
        #print("End of training epoch {} of training; got log keys: {}".format(epoch, keys))
        print("Epoch {}/{} > ".format(epoch+1, epochs))
        #if (epoch+1) == epochs:
        #    print('')

# Checkpointing & Early Stopping
if batch_size > 1:
    metric = 'binary_accuracy'
else:
    metric = 'accuracy'
mon = 'val_' +metric
es = EarlyStopping(monitor=mon, patience=10, restore_best_weights=True) # Temporarily turning this off because I want to observe the full scope
checkpointname = "/scratch/mssric004/Checkpoints/testing-{epoch:02d}.ckpt"
localcheck = "/scratch/mssric004/TrueChecks/" + modelname +".ckpt"
be = ModelCheckpoint(localcheck, monitor=mon, mode='auto', verbose=2, save_weights_only=True, save_best_only=True)
mc = ModelCheckpoint(checkpointname, monitor=mon, mode='auto', verbose=2, save_weights_only=True, save_best_only=False) #Maybe change to true so we can more easily access the "best" epoch

# Slice generation stuff
optim = keras.optimizers.Adam(learning_rate=0.0001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later
channels = 1 # Replicating into 3 channels is proving annoying

def generatePriorityModels(slices, metric):
    models = []
    weights = []
    for i in range(len(slices)):
        global n
        n = slices[i]
        print("Fitting for slice", n, ".")
        display = [str(x) for x in slices]
        display.insert(i, "->")
        print(display)
        # Set up a model
        model = gen_advanced_2d_model(w, h, channels, classes=classNo)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy']) #metrics=[tf.keras.metrics.BinaryAccuracy()]
        models.append(model)
        weight = model.get_weights()
        weights.append(weight)
    return models, weights

# Build model list
model_list, initials = generatePriorityModels(priority_slices, metric)

def reset_weights(reused_model, init_weights):
    reused_model.set_weights(init_weights)

# Define voting
def soft_voting(predicted_probas : list, weights : list) -> np.array:

    #sv_predicted_proba = np.mean(predicted_probas, axis=0)
    sv_predicted_proba = np.average(predicted_probas, axis=0, weights=weights)
    sv_predicted_proba[:,-1] = 1 - np.sum(sv_predicted_proba[:,:-1], axis=1)    

    return sv_predicted_proba, sv_predicted_proba.argmax(axis=1)

# K-Fold setup
n_folds = 5
if testing_mode:
    n_folds = 2
acc_per_fold = []
loss_per_fold = []
rar = 0
skf = StratifiedKFold(n_splits=n_folds, random_state=rar, shuffle=True)
mis_classes = []

print("\nStarting cross-fold validation process...")
print("Params:", epochs, "epochs &", batch_size, "batches.")
fold = 0
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

    print("Setting up dataloaders...")
    # TO-DO: Augmentation stuff
    # Data loaders
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_set = (
        train.shuffle(len(train))
        .map(load_slice_wrapper)
        .batch(batch_size)
        .map(fix_shape)
        .prefetch(batch_size)
    )

    # Only rescale.
    validation_set = (
        val.shuffle(len(x_val))
        .map(load_slice_wrapper)
        .batch(batch_size)
        .map(fix_shape)
        .prefetch(batch_size)
    )

    # Run the model
    #if fold == 1:
    #    model.summary()

    # Reset the weights
    print("Resetting weights...")
    for i in range(len(model_list)):
        reset_weights(model_list[i], initials[i])

    print("Fitting model...")
    voting_weights = []
    names = []
    slices = priority_slices
    for i in range(len(slices)):
        global n
        n = slices[i]
        print("Fitting for slice", n, ".")
        display = [str(x) for x in slices]
        display.insert(i, "->")
        print(display)
        if testing_mode:
        #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0)
            history = model_list[i].fit(train_set, validation_data=validation_set, epochs=epochs, verbose=0, callbacks=[be, CustomCallback()]) # DON'T SPECIFY BATCH SIZE, CAUSE INPUT IS ALREADY A BATCHED DATASET
        else:
        #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0, shuffle=True)
            history = model_list[i].fit(train_set, validation_data=validation_set, epochs=epochs, callbacks=[be, es], verbose=0, shuffle=True)
            # Not saving checkpoints FOR NOW
        print(history.history)
        model_list[i].load_weights(localcheck)
        val_weight = history.history['val_accuracy'][-1]
        names.append("model"+str(n))
        voting_weights.append(val_weight)
        # Clean up checkpoints
        import glob
        found = glob.glob(localcheck+"*")
        removecount = 0
        for checkfile in found:
            removecount += 1
            os.remove(checkfile)

    print("RESULTS FOR FOLD", fold, ":")
    
    # Final evaluation
    preds=[]
    predi=[]
    evals=[]
    model_loss=[]
    print("Evaluating...")
    for j in range(len(model_list)):
        n = priority_slices[j]
        scores = model_list[j].evaluate(test_set, verbose=0)
        acc = scores[1]*100
        loss = scores[0]
        model_loss.append(loss)
        evals.append(acc)
        #try:
        print("Predicting on model in rank", j, ":")
        #print(model_list[j].summary())
        #print("Example shape of what we are using to predict:", next(iter(test_set_x)).shape)
        pred = model_list[j].predict(test_set_x)
        preds.append(pred)
        predi.append(np.argmax(pred, axis=1))
        #except:
        #    preds.append[[-1,-1]]
        #    predi.append(-1)
    sv_predicted_proba, sv_predictions = soft_voting(preds, voting_weights)
    Y_test=np.argmax(y_test, axis=1)
    for k in range(len(model_list)):
        print(f"Accuracy of {names[k]}: {accuracy_score(Y_test, predi[k])}")
    acc = accuracy_score(Y_test, sv_predictions)*100
    print("Fold", fold, "evaluated scores - Soft Voting Acc:", acc, "Loss:", loss)
    acc_per_fold.append(acc)
    loss_per_fold.append(mean(model_loss))
    #loss_per_fold.append(loss)
    print("Average so far:" , np.mean(acc_per_fold), "+-", np.std(acc_per_fold))

# Save outside the loop my sire
'''
if testing_mode:
    modelname = "ADModel_K_Testing"
modelname = modelname +".h5"
'''
#model.save(modelname)
# Electing not to save for now since the file it generates is HUGE
loc = "Means/" + logname
print("Saving means to:", loc)
np.savez(loc, acc_per_fold, loss_per_fold)

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
if not testing_mode:
    loc = "Means/" + logname
    np.savez(loc, acc_per_fold, loss_per_fold)

'''
# Check those tricky images
mis_d = Counter(mis_classes)
howmany = 5
print("\nMismatched test images (top ", howmany, "):", sep='')
common = mis_d.most_common()
for j in range(0, howmany-1):
    print(common[j])
'''
print("Done.")