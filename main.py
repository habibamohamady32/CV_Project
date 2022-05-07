import cv2
from Kmeans_task import *

k= 3
img = cv2.imread("Image_to_be_segmented.jpg")
cv2.imwrite(f"images/SegmantedImage.jpg", reducedColors_image(img,k))