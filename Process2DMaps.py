import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
from PIL import Image
print("Imports done.")

# 179, 208, 179

root = "Grad-Maps-2D\\relu\\"
path = root + "attention_map_0_0_0.png"
refloc = "Grad-Maps-2D\\reference.jpg"

grad = Image.open(path)
ref = Image.open(refloc)
print("Grad:", grad.size)
print("Ref:", ref.size)

grad = grad.convert("RGBA")
ref = ref.convert("RGBA")
ref = ref.resize((612, 495))

new_img = Image.blend(ref, grad, 0.5)
new_img.save("Grad-Maps-2D\\Maps\\test.png","PNG")
ref.save("Grad-Maps-2D\\Maps\\orig.png","PNG")

print("All done!")

