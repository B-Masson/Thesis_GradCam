import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
print("Imports done.")

# 208, 240, 256

path = "Grad-Maps-2D\\relu"
direct = os.listdir(path)
count = 0
for im in direct:
    if count < 1:
        # Read Images
        img = mpimg.imread(op.join(path, im))
        # Output Images
        print("Shape of map image:", img.shape)
        plt.imshow(img, interpolation='nearest')
        plt.show()
        count += 1
        
print("Excess:", (412-208)+(476-240))

print("All done.")