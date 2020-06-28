import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data/yes/Y1.jpg')
w, h, ch = img.shape
# w, h = 10, 10
X = []
Y = []

offset = 10

for row in np.arange(0, h, offset):
    for col in np.arange(0, w, offset):
        X.append(row)
        Y.append(col)
        cv2.circle(img, (row, col), 1, (0, 0, 255), -1)


print(X)
print(Y)

cv2.imshow('Uni', img)

cv2.waitKey(0)
# plt.scatter(X, Y)
# plt.show()
