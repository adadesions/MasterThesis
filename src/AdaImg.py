from skimage import io
import cv2
import numpy as np
import plotly.express as px

class AdaImg:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_obj = None
        self.img_gray = None
        self.lowerRes_imgs = []
        self.gaborParams = {
            'ksize': (50, 50),
            'sigma': 3.0,
            'theta': np.pi/4,
            'lambd': np.pi/4,
            'gamma': 0.5,
            'psi': 0,
            'ktype': cv2.CV_32F
        }
        self.gabor_kernel = None
        # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
        # ksize - size of gabor filter (n, n)
        # sigma - standard deviation of the gaussian function
        # theta - orientation of the normal to the parallel stripes
        # lambd - wavelength of the sunusoidal factor
        # gamma - spatial aspect ratio
        # psi - phase offset
        # ktype - type and range of values that each pixel in the gabor kernel can hold


    def show(self):
        cv2.waitKey()

    def set_imgPath(self, img_path):
        self.img_path = img_path


    def readImg(self, img_path=""):
        if (img_path != ""):
            self.img_path = img_path

        print("file path:", self.img_path)
        self.img_obj = cv2.imread(self.img_path)
        self.img_gray = cv2.cvtColor(self.img_obj, cv2.COLOR_BGR2GRAY)


    def showImg(self):
        fig = px.imshow(self.img_obj)
        fig.show()


    def calLowerRes(self, levels):
        temp = self.img_obj.copy()
        self.lowerRes_imgs.append(temp)

        for lv in range(levels):
            temp = cv2.pyrDown(temp)
            (h, w, ch) = temp.shape
            
            if (w <= 1 and h <= 1):
                break

            self.lowerRes_imgs.append(temp)
        
        print("=== Done PyramidDown, len: {} ===".format(len(self.lowerRes_imgs)))


    def getLowerRes(self):
        return self.lowerRes_imgs


    def lowerResInfo(self):
        for idx, img in enumerate(self.lowerRes_imgs):
            print(idx, img.shape)


    def imShowLowerRes(self):
        winName = "LowRes"
        for idx, lowImg in enumerate(self.lowerRes_imgs):
            winNameIdx = "-".join([winName, str(idx)])
            cv2.imshow(winNameIdx, lowImg)

    def setGaborKernel(self):
        self.gabor_kernel = cv2.getGaborKernel(**self.gaborParams)


    def setGaborParams(self, newParams):
        gaborKey = ('ksize', 'sigma', 'theta', 'lambd', 'gamma', 'psi', 'ktype')
        for key in gaborKey:
            if key in newParams.keys():
                continue
            else:
                print('newParams missing {} key', key)
                return -1
        
        self.gaborParams = newParams


    def showGaborKernel(self, scale):
        h, w = self.gabor_kernel.shape[:2]
        scaledKernel = cv2.resize(self.gabor_kernel, (scale*w, scale*h), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Gabor Kernel", scaledKernel)


    def getGaborResponse(self):
        self.setGaborKernel()

        img_filtered = cv2.filter2D(self.img_gray, cv2.CV_8UC3, self.gabor_kernel)

        cv2.imshow('img_gray', self.img_gray)
        cv2.imshow('img_filtered', img_filtered)


