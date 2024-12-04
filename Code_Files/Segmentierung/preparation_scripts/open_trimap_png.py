from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Open the image file
path = r'path'
# path = r'C:\Users\janku\Documents\ShareX\Screenshots\2024-11\python_qtwX4Jtzex.png'
img = Image.open(path)
# img = img.resize((120, 120))

# Display the image
plt.imshow(img)
plt.show()
