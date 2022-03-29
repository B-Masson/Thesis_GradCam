# Generate an image in which a GradCam map is overlayed with the original image
from PIL import Image
import cv2

# Parameters
index = 0
img_target = "Grad-Originals\\brain_slice_"+str(index)+".png"
grad_target = "Grad-Maps\\layer4\\attention_map_"+str(index)+"_0_0.png"

img = Image.open(img_target)
grad = Image.open(grad_target)
print("Base image:", img.size, " | Mode:", img.mode)
print("Grad Map:", grad.size, " | Mode:", grad.mode)
print("Same dimensions:", (img.size == grad.size))

# Ensure images are the same
if img.size != grad.size:
    print("Resizing base image...")
    img = img.resize(grad.size)

if img.mode != grad.mode:
    print("Converting base image mode...")
    img = img.convert(grad.mode)

# Overlay
trans = 0.5
print("Overlaying images at", trans, "transparency...")
map = Image.blend(img, grad, alpha=trans)
map.save("gradmap.jpg")

print("All done.")