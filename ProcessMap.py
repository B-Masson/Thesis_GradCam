import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import NIFTI_Engine as ne

root = "Grad-Maps\\pool"
loc = root + "\\attention_map_0_0_0.nii.gz"

# Load in nifti
image = nib.load(loc)
data = image.get_fdata(dtype='float32')
print("Map loaded in.")

# Rescale
#data = np.asarray(data)
#data = ne.resizeADNI(data, stripped=True)
#print(data.shape)
rescale = resample_to_output(image, voxel_sizes=(208, 240, 256))
print("Map rescaled.")

# Save
outfile = root + "\\grad_map_scaled_0.nii.gz"
rescale.to_filename(outfile)
#nib.Nifti1Image(rescale, image.affine).to_filename(outfile)
print("New rescaled map saved. All done!")