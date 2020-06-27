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
# Set new gabor params
newGaborParams = {
    'ksize': (64, 64),
    'sigma': 5.0,
    'theta': np.pi/4,
    'lambd': np.pi,
    'gamma': 1,
    'psi': 0,
    'ktype': cv2.CV_32F
}
ada.setGaborParams(newGaborParams)
ada.getGaborFusion()
ada.getGaborResponse()

# TODO: Hilbert Curve Transform
# TODO: Classification by ML

ada.show()