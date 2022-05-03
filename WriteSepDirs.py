# Code to generate two text files, one list of image paths for (currently) the ADNI dataset, and one list of their corresponding labels
import os
import os.path as op
import re

from numpy import not_equal

# https://stackoverflow.com/questions/71045309/how-to-create-a-dataset-for-tensorflow-from-a-txt-file-containing-paths-and-labe

gen_tests = False
tiny_mode = True
strip = False
norm = False
if gen_tests:
	root = "Data\\ADNI_Test"
elif tiny_mode:
    root = "Data\\ADNI_Test_Single"
else:
	root = "Data\\ADNI"

# Set this to generate the different sets
mode = 1
if gen_tests:
    filename = "Directories/test_adni_" + str(mode)
elif tiny_mode:
    filename = "Directories/test_tiny_adni_" +str(mode)
else:
	filename = "Directories/adni_" + str(mode)

if norm:
    imgfile = filename+"_images_normed.txt"
    labelfile = filename+"_labels_normed.txt"
elif strip:
    imgfile = filename+"_images_stripped.txt"
    labelfile = filename+"_labels_stripped.txt"
else:
    imgfile = filename + "_images.txt"
    labelfile = filename + "_labels.txt"
print("Writing to", filename)
newline = '' # Little trick to avoid having a newline at the end of file
countCN = 0
countMCI = 0
countAD = 0
write_count = 0
label_count = 0
blanks = 0

with open(imgfile, 'w') as i:
    with open(labelfile, 'w') as l:
        class_dirs = os.listdir(root)
        exclusions = []
        if mode == 1:
            exclusions.append("AD")
            print("Mode: Classify CN vs. MCI")
        elif mode == 2:
            exclusions.append("MCI")
            print("Mode: Classify CN vs. AD")
        elif mode == 4:
            exclusions.append("CN")
            exclusions.append("MCI")
            print("Mode: ONLY AD. Only use for testing purposes.")
        elif mode == 5:
            exclusions.append("CN")
            exclusions.append("AD")
        else:
            print("Mode: Classify from all 3 categories.")
        print("\nReading through image directory...\n")
        for classes in class_dirs:
            if classes not in exclusions and classes != "Zips":
                for scan in os.listdir(op.join(root, classes)):
                    for type in os.listdir(op.join(root, classes, scan)):
                        for date in os.listdir(op.join(root, classes, scan, type)):
                            for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
                                image_root = op.join(root, classes, scan, type, date, image_folder)
                                image_options = os.listdir(image_root)
                                image_file = '[none]'
                                if norm:
                                    for file in image_options:
                                        if "NORMED" in file:
                                            image_file = file
                                elif strip:
                                    for file in image_options:
                                        if "STRIPPED" in file:
                                            image_file = file
                                else:
                                    for file in image_options:
                                        if "NORMED" not in file:
                                            if "STRIPPED" not in file:
                                                image_file = file
                                if gen_tests or tiny_mode:
                                    print("Code chose:", image_file)
                                if image_file == '[none]':
                                    blanks += 1
                                else:
                                    image_dir = op.join(image_root, image_file)
                                    i.write(newline+image_dir)
                                    write_count += 1
                                    if classes == "CN":
                                        l.write(newline+"0")
                                        label_count += 1
                                        countCN += 1
                                    elif classes == "MCI":
                                        l.write(newline+"1")
                                        label_count += 1
                                        countMCI += 1
                                    elif classes == "AD":
                                        if mode == 2:
                                            l.write(newline+"1")
                                        else:
                                            l.write(newline+"2")
                                        label_count += 1
                                        countAD += 1
                                    else:
                                        print("One of the folders does not match any of the expected forms.")
                                    newline = '\n'

print(write_count, "images written!")
print(label_count, "labels written!")
counts = {"CN": countCN, "MCI": countMCI, "AD": countAD}
for cata in counts:
    if (counts[cata] != 0):
        print(cata, ": ", counts[cata], " images.", sep='')
if blanks != 0:
    print("Skipped over", blanks, "entries that did not match specifications.")