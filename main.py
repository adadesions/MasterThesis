import cv2
import numpy as np
from src.AdaImg import AdaImg

# Read image
ada = AdaImg('data/yes/Y1.jpg')
# ada = AdaImg('data/no/N2.jpg')
ada.readImg()
# ada.showImg()

# Pyramid method
ada.calLowerRes(levels=10)
ada.lowerResInfo()
# ada.imShowLowerRes()

# Gabor function and get response from any parametize
ada.getGaborResponse()
# Show Gabor Kernel
ada.showGaborKernel(scale=3)

# Set new gabor params
newGaborParams = {
    'ksize': (4, 4),
    'sigma': 3.0,
    'theta': np.pi/4,
    'lambd': np.pi/4,
    'gamma': 0,
    'psi': 0,
    'ktype': cv2.CV_32F
}
ada.setGaborParams(newGaborParams)
ada.getGaborResponse()
ada.showGaborKernel(scale=10)

# TODO: Hilbert Curve Transform
# TODO: Classification by ML

ada.show()