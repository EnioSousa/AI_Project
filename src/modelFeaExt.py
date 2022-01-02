#importing the required libraries
import numpy as np
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
import matplotlib.pyplot as plt

#reading the image 
#image = imread('data/test/cats/cat.3.jpg',as_gray=True)
image = imread('data/test/dogs/dog.5.jpg',as_gray=True)

a = prewitt_h(image)
b = prewitt_v(image)
c = prewitt_v(a)

#imshow(a, cmap="gray")
#imshow(b, cmap="gray")
imshow(b, cmap="gray")

plt.show()