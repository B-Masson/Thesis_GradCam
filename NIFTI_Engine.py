# Host of functions for use in data collection and pre-processing for the brain scans
# Richard Masson
import numpy as np
import skimage, os
import os.path as op
#import scipy.misc
from skimage.io import imread
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
#from tqdm import tqdm
#from time import sleep
#import tensorflow as tf

# Function: Run through an entire data folder and extract all data of a certain scan type. Graps NIFTI data and returns numpy arrays for the x-array
# Also returns an an array alongside with the patient ID and the day of the MRI
# Orientation currently sets the choice orientation in which 2D reference images are generates (not: the returned data is still 3D)
def extractArrays(scantype, w, h, d, orientation=0, root="C:\\Users\\richa\\Documents\\Uni\\Thesis\\central.xnat.org"):
    scan_array = []
    meta_array = []
    min_intensity = 500
    max_intensity = -1
    sample_dirs = os.listdir(root)
    print("Loading in scan data...")
    for sample in sample_dirs:
        if scantype == "all":
            for scan in os.listdir(op.join(root, sample)):
                if ("anat" in scan) or ("func" in scan): #Mostly to sift out BIDS folders/random unwanted stuff
                    image_root = op.join(root, sample, scan, 'NIFTI')
                    #print("Reading file at", image_root, ": ", os.listdir(image_root)[0])
                    image_file = os.listdir(image_root)[0]
                    image_dir = op.join(image_root, image_file)
                    image_data_raw = nib.load(image_dir).get_fdata()
                    image_data = organiseImage(image_data_raw, w, h, d)
                    scan_array.append(image_data)
                    meta_segments = sample.split("_")
                    meta_array.append({'ID': meta_segments[0], 'day': int(meta_segments[2][1:])})
                    try:
                        if orientation == 1:
                            plt.imshow(image_data[50,:,:], cmap='bone')
                            plt.savefig("a sample_1.png")
                        elif orientation == 2:
                            plt.imshow(image_data[:,50,:], cmap='bone')
                            plt.savefig("a_sample_2.png")
                        elif orientation == 3:
                            plt.imshow(image_data[:,:,30], cmap='bone')
                            plt.savefig("a_sample_3.png")
                    except TypeError as e:
                        print("Cannot display example slices:", e)
        elif scantype in os.listdir(op.join(root, sample)): # This needs to be cleaned up and incorporated into the above at some point, to reduce redundant code
            root_niftis = op.join(root, sample, scantype)
            check_niftis = os.listdir(root_niftis)
            if "NIFTI" in check_niftis:
                image_root = op.join(root_niftis, 'NIFTI')
                image_file = os.listdir(image_root)[0]
                image_dir = op.join(image_root, image_file)
                image_data_raw = nib.load(image_dir).get_fdata()
                image_data = organiseImage(image_data_raw, w, h, d)
                scan_array.append(image_data)
                meta_segments = sample.split("_")
                meta_array.append({'ID': meta_segments[0], 'day': int(meta_segments[2][1:])})
                try:
                    if orientation == 1:
                        plt.imshow(image_data[50,:,:], cmap='bone')
                        plt.savefig("a sample_1.png")
                    elif orientation == 2:
                        plt.imshow(image_data[:,50,:], cmap='bone')
                        plt.savefig("a_sample_2.png")
                    elif orientation == 3:
                        plt.imshow(image_data[:,:,30], cmap='bone')
                        plt.savefig("a_sample_3.png")
                except TypeError as e:
                    print("Cannot display example slices:", e)
            else: print("No NIFTI file found. This should not occur if the dataset is half-decent.")
        else:
            print("Warning:", sample, "does not possess data of type", scantype)
    return scan_array, meta_array

def extractADNI(w, h, d, orientation=0, root="C:\\Users\\richa\\Documents\\Uni\\Thesis\\central.xnat.org", mode=3):
    scan_array = []
    label_array = []
    class_array = []
    class_dirs = os.listdir(root)
    exclusions = "na"
    ADcounter = 0 # No. of incorrect file types found in AD folder
    Xcounter = 0 # No. of incorrect file types outside AD
    if mode == 1:
        exclusions = "AD"
        print("Mode: Classify CN vs. MCI")
    elif mode == 2:
        exclusions = "MCI"
        print("Mode: Classify CN vs. AD")
    else:
        print("Mode: Classify from all 3 categories.")
    print("\nLoading in scan data...\n")
    for classes in class_dirs:
        if classes != exclusions and classes != "Zips":
            for scan in os.listdir(op.join(root, classes)):
                for type in os.listdir(op.join(root, classes, scan)):
                    for date in os.listdir(op.join(root, classes, scan, type)):
                        for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
                            image_root = op.join(root, classes, scan, type, date, image_folder)
                            image_file = os.listdir(image_root)[0]
                            image_dir = op.join(image_root, image_file)
                            #print("Reading file at", image_dir)
                            image_data_raw = nib.load(image_dir).get_fdata()
                            image_data = organiseADNI(image_data_raw, w, h, d)
                            scan_array.append(image_data)
                            if classes == "CN":
                                label_array.append(0)
                                #print("Setting label to 0 (CN)")
                            elif classes == "MCI":
                                label_array.append(1)
                                #print("Setting label to 1 (MCI)")
                            elif classes == "AD":
                                label_array.append(2)
                                #print("Setting label to 2 (AD")
                            else:
                                print("One of the folders does not match any of the expected forms.")
                            '''
                            try:
                                image_data_raw = nib.load(image_dir).get_fdata()
                                image_data = organiseADNI(image_data_raw, w, h, d)
                                scan_array.append(image_data)
                                if classes == "CN":
                                    label_array.append(0)
                                elif classes == "MCI":
                                    label_array.append(1)
                                elif classes == "AD":
                                    label_array.append(2)
                                else:
                                    print("One of the folders does not match any of the expected forms.")
                            except:
                                if classes == "AD":
                                    ADcounter += 1
                                else:
                                    Xcounter += 1
                            '''
                            try:
                                if orientation == 1:
                                    plt.imshow(image_data[50,:,:], cmap='bone')
                                    plt.savefig("ADNI_1.png")
                                elif orientation == 2:
                                    plt.imshow(image_data[:,50,:], cmap='bone')
                                    plt.savefig("ADNI_2.png")
                                elif orientation == 3:
                                    plt.imshow(image_data[:,:,30], cmap='bone')
                                    plt.savefig("ADNI_3.png")
                            except TypeError as e:
                                print("Cannot display example slices:", e)
                            class_array.append(classes)
    print("Samples with the wrong data type: AD[", ADcounter, " ] Else[", Xcounter, " ]")
    return scan_array, label_array

def extractADNILoader(w, h, d, orientation=0, root="C:\\Users\\richa\\Documents\\Uni\\Thesis\\central.xnat.org", mode=3):
    scan_array = []
    label_array = []
    class_array = []
    class_dirs = os.listdir(root)
    exclusions = "na"
    if mode == 1:
        exclusions = "AD"
        print("Mode: Classify CN vs. MCI")
    elif mode == 2:
        exclusions = "MCI"
        print("Mode: Classify CN vs. AD")
    else:
        print("Mode: Classify from all 3 categories.")
    print("\nLoading in scan data...\n")
    for classes in class_dirs:
        if classes != exclusions and classes != "Zips":
            for scan in os.listdir(op.join(root, classes)):
                for type in os.listdir(op.join(root, classes, scan)):
                    for date in os.listdir(op.join(root, classes, scan, type)):
                        for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
                            image_root = op.join(root, classes, scan, type, date, image_folder)
                            image_file = os.listdir(image_root)[0]
                            image_dir = op.join(image_root, image_file)
                            image_data_raw = nib.load(image_dir).get_fdata()
                            image_data = organiseADNI(image_data_raw, w, h, d)
                            scan_array.append(image_data)
                            if classes == "CN":
                                label_array.append(0)
                            elif classes == "MCI":
                                label_array.append(1)
                            elif classes == "AD":
                                label_array.append(2)
                            else:
                                print("One of the folders does not match any of the expected forms.")
                            class_array.append(classes)
    scan_array = np.asarray(scan_array)
    label_array = np.asarray(label_array)
    print(scan_array[0].shape)
    dataset = tf.data.Dataset.from_tensor_slices((scan_array, label_array))
    return dataset

def extractSingle(w, h, d, root, type):
    types = os.listdir(root)
    dates = os.listdir(op.join(root, types[0]))
    folders = os.listdir(op.join(root, types[0], dates[0]))
    images = os.listdir(op.join(root, types[0], dates[0], folders[0]))
    true_root = op.join(root, types[0], dates[0], folders[0], images[0])
    scan_array = []
    label_array = []
    image_data_raw = nib.load(true_root).get_fdata()
    image_data = organiseADNI(image_data_raw, w, h, d)
    scan_array.append(image_data)
    if type == "CN":
        label_array.append(0)
    elif type == "MCI":
        label_array.append(1)
    elif type == "AD":
        label_array.append(2)
    else:
        print("Incorrect class inputted.")
    return scan_array, label_array

def extractSlices(w, h, d, root, orientation, exclusions="na", limit=-1):
    scan_array = []
    label_array = []
    mid = 0
    class_dirs = os.listdir(root)
    count = 0
    print("\nLoading in scan data...\n")
    for classes in class_dirs:
        if classes != exclusions and classes != "Zips":
            for scan in os.listdir(op.join(root, classes)):
                for type in os.listdir(op.join(root, classes, scan)):
                    for date in os.listdir(op.join(root, classes, scan, type)):
                        for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
                            image_root = op.join(root, classes, scan, type, date, image_folder)
                            image_file = os.listdir(image_root)[0]
                            image_dir = op.join(image_root, image_file)
                            image_data_raw = nib.load(image_dir).get_fdata()
                            print("Raw shape:", image_data_raw.shape)
                            image_data = organiseADNI(image_data_raw, w, h, d)
                            count += 1
                            #print(count, "...", sep='', end='')
                            #print("Shape:", image_data.shape)
                            if orientation == 1:
                                mid = (int)(image_data.shape[0]/2)
                                #print("Midpoint is", mid)
                                #print("Appended x element:", image_data[:,:,mid].shape)
                                #print("Original shape:", image_data.shape)
                                sliced = image_data[mid,:,:]
                                sliced = np.array(sliced)
                                #print("Slice shape:", sliced.shape)
                                sliced = np.squeeze(sliced)
                                #print("Squeeze shape:", sliced.shape)
                                # Try make this a 3-channel image
                                img = cv2.merge((sliced,sliced,sliced))
                                #print("3-channel shape:", img.shape)
                                img = np.transpose(img, (2, 0, 1))
                                #print("Reordered shape:", img.shape)
                                scan_array.append(img)
                            elif orientation == 2:
                                mid = (int)(image_data.shape[1]/2)
                                #print("Midpoint is", mid)
                                #print("Appended x element:", image_data[:,:,mid].shape)
                                #print("Original shape:", image_data.shape)
                                sliced = image_data[:,mid,:]
                                sliced = np.array(sliced)
                                #print("Slice shape:", sliced.shape)
                                sliced = np.squeeze(sliced)
                                #print("Squeeze shape:", sliced.shape)
                                # Try make this a 3-channel image
                                img = cv2.merge((sliced,sliced,sliced))
                                #print("3-channel shape:", img.shape)
                                img = np.transpose(img, (2, 0, 1))
                                #print("Reordered shape:", img.shape)
                                scan_array.append(img)
                            elif orientation == 3:
                                mid = (int)(image_data.shape[2]/2)
                                #print("Midpoint is", mid)
                                #print("Appended x element:", image_data[:,:,mid].shape)
                                #print("Original shape:", image_data.shape)
                                sliced = image_data[:,:,mid]
                                sliced = np.array(sliced)
                                #print("Slice shape:", sliced.shape)
                                sliced = np.squeeze(sliced)
                                #print("Squeeze shape:", sliced.shape)
                                # Try make this a 3-channel image
                                img = cv2.merge((sliced,sliced,sliced))                               
                                #print("3-channel shape:", img.shape)
                                imagename = "Grad-Originals\\brain_slice_"+str(count-1)+".png"
                                plt.imsave(imagename, img, cmap='bone')
                                img = np.transpose(img, (2, 0, 1))
                                #print("Reordered shape:", img.shape)
                                scan_array.append(img)
                            if classes == "CN":
                                label_array.append(0)
                            elif classes == "MCI":
                                label_array.append(1)
                            elif classes == "AD":
                                label_array.append(2)
                            else:
                                print("One of the folders does not match any of the expected forms.")
                            if limit != -1:
                                if count >= limit:
                                    print("Cutting off at limit (", limit, ")", sep='')
                                    return scan_array, label_array
    #print("")
    return scan_array, label_array

# Normalise pixel values to range from 0 to 1:
# Need to run AD_MinMax to get min and max values.
def normalize(image_data, min=0, max=4095):
    image_data = (image_data - min) / (max - min)
    image_data = image_data.astype("float32")
    return image_data

def normalize_old(image_data, min=-1000, max=400):
    image_data[image_data < min] = min
    image_data[image_data > max] = max
    image_data = (image_data - min) / (max - min)
    image_data = image_data.astype("float32")
    return image_data

# Resize the data to some uniform amount so it actually fits into a training model
def resize(image_data, w=128, h=128, d=64):
    # Get current dimensions
    width = image_data.shape[0]
    height = image_data.shape[1]
    depth = image_data.shape[-1]
    # Compute all the factors that we need to scale the dimensions by
    width_factor = 1/(width/w)
    height_factor = 1/(height/h)
    depth_factor = 1/(depth/d)
    #length_factor = 1
    # Resize using factors
    image_data = ndimage.zoom(image_data, (width_factor, height_factor, depth_factor), order=1)
    return image_data

# Not sure what version this is. Possibly for the ADNI data?
def resizeADNI(image_data, w = 208, h = 240, d = 256):
    # Get current dimensions
    width = image_data.shape[0]
    height = image_data.shape[1]
    depth = image_data.shape[2]
    # Compute all the factors that we need to scale the dimensions by
    width_factor = 1/(width/w)
    height_factor = 1/(height/h)
    depth_factor = 1/(depth/d)
    length_factor = 1
    # Resize using factors
    image_data = ndimage.zoom(image_data, (width_factor, height_factor, depth_factor, length_factor), order=1)
    #image_data = ndimage.zoom(image_data, (width_factor, height_factor, depth_factor), order=1)
    return image_data

def organiseImage(data, w, h, d):
    data = normalize(data)
    data = resize(data, w, h, d)
    return data

def organiseADNI(data, w, h, d):
    data = normalize(data)
    #print("ADNI dimensions by default:", data.shape)
    data = resizeADNI(data, w, h, d)
    return data

#if __name__ == "__main__":
    #Why does this trigger even when its not main?
    #scans, meta = extractArrays('anat3', 0) #Orientation 0 to not display anything
    #print(meta)