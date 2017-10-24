import numpy as np
import os
from matplotlib import pyplot as plot
from scipy.ndimage import imread
from math import floor


class Image(object) :
    def __init__(self) :
        self._image = None
        self.type = None
        self.dimensions = []

    def load(self,path,mode = 'L') :
        try :
            self._image = imread(os.path.abspath(path), mode = mode)
        except :
            print("Error! Could not read the image from the path specified: %s"%os.path.abspath(path))
            return
        try :
            self._image = np.asarray(self._image, dtype = float)
            self.dimensions = self._image.shape
            self.type = path.split(".")[-1]
        except :
            print("Internal Error! Image file not supported")

    def getImg(self):
        return self._image

    def setImg(self, image) :
        image = np.asarray(image, dtype = float)
        if len(image.shape) == 2 :
            self._image = image
            try :
                self.dimensions = self._image.shape
            except :
                print("Internal Error! Image file not supported")
        else :
            print("Assignment Error. Given input is not an image")

    imarray = property(getImg,setImg)

    def show(self, mode='Greys_r', name=None) :
        try :
            plot.imshow(self._image,cmap=mode)
        except :
            print("Image Could not be displayed")
            return
        if not name is None :
            plot.title(name)
            name = "images/"+name
            plot.imsave(name,self._image,cmap=mode)
        plot.show()

    def convolve(self,mask):
        mask = np.asarray(mask,dtype = float)
        if len(mask.shape) != 2:
            print("Invalid Mask. Please input a 2D Mask")
            return
        (m,n) = mask.shape
        padHeight = int(floor(m/2))
        padWidth = int(floor(n/2))
        (M,N) = self.dimensions
        img = np.ones((M+padHeight*2,N+padWidth*2))*128
        new_img = np.ones((M+padHeight*2,N+padWidth*2))
        img[padHeight:-padHeight,padWidth:-padWidth] = self._image

        for i in range(padHeight,M+padHeight):
            for j in range(padWidth,N+padWidth):
                new_img[i,j]=sum(sum(img[i-padHeight:i+m-padHeight,j-padWidth:j+n-padWidth]*mask))

        return new_img[padHeight:-padHeight,padWidth:-padWidth]
