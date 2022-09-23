import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import NIFTI_Engine as ne

root = "Grad-Maps\\relu"
loc = root + "\\attention_map_0_0_0.nii.gz"

# Load in nifti
image = nib.load(loc)
data = image.get_fdata(dtype='float32')
print("Map loaded in.")

slicenum = 58
sliced = data[:,slicenum,:]
'''
from PIL import Image
grad = Image.fromarray(np.uint8(sliced)).convert('RGBA')
grad.save("Grad-Maps\\Maps\\test.png","PNG")

rgbimg = Image.new("RGBA", grad.size)
rgbimg.paste(grad)
rgbimg.save("Grad-Maps\\Maps\\rgb.png","PNG")
'''
import matplotlib.pyplot as plt
maptype = 'plasma'
plt.imshow(sliced, cmap=maptype)
plt.savefig("Grad-Maps\\Maps\\"+maptype+".png")
print("Saved a", maptype, "map.")

print("All done!")