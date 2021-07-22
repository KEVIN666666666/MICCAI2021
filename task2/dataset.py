from matplotlib import pyplot as plt
from matplotlib import image as Image
import numpy as np
import os

PATH = "../pre_data_V1/"
image_names = os.listdir(PATH)

data = []
for image_name in image_names:
    if not image_name.endswith(".jpg"):
        continue  # not a jpg file
    image = Image.imread(PATH + image_name)  # RGB mode
    data.append(image)


plt.imshow(image)
plt.show()

print("Finish.")
