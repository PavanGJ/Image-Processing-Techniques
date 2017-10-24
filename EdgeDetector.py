import numpy as np
import Image as im

class LaplaceofGaussian(object):
    def __init__(self):
        self.mask = np.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]])
        self.gaussian_filter = np.array([[1/16.,1/8.,1/16.],[1/8.,1/4.,1/8.],[1/16.,1/8.,1/16.]])
        self.sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        self.sobel_y = np.fliplr(self.sobel_x).transpose()

    def loadImage(self,image):
        self.img = im.Image()
        self.img.load(image)

    def performLOG(self):
        cimg = im.Image()
        self.img.imarray = self.img.convolve(self.gaussian_filter)
        G_x = self.img.convolve(self.sobel_x)
        G_y = self.img.convolve(self.sobel_y)
        G = pow((G_x*G_x + G_y*G_y),0.5)
        G = (G>32)*G
        temp_img = self.img.convolve(self.mask)
        if (temp_img == None).any():
            return
        (M,N) = temp_img.shape
        #detect zero crossing by checking values across 8-neighbors on a 3x3 grid
        temp = np.zeros((M+2,N+2))
        temp[1:-1,1:-1] = temp_img
        img = np.zeros((M,N))
        for i in range(1,M+1):
            for j in range(1,N+1):
                if temp[i,j]<0:
                    for x,y in (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1):
                            if temp[i+x,j+y]>0:
                                img[i-1,j-1] = 1
        cimg.imarray = np.logical_and(img,G)

        cimg.show(name = "LOGSobel.jpg")

class DiffofGaussian(object):
    def __init__(self):
        self.mask = np.array([[0,0,-1,-1,-1,0,0],[0,-2,-3,-3,-3,-2,0],[-1,-3,5,5,5,-3,-1],[-1,-3,5,16,5,-3,-1],[-1,-3,5,5,5,-3,-1],[0,-2,-3,-3,-3,-2,0],[0,0,-1,-1,-1,0,0]])
        self.gaussian_filter = np.array([[1/16.,1/8.,1/16.],[1/8.,1/4.,1/8.],[1/16.,1/8.,1/16.]])
        self.sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        self.sobel_y = np.fliplr(self.sobel_x).transpose()

    def loadImage(self,image):
        self.img = im.Image()
        self.img.load(image)

    def performDOG(self):
        cimg = im.Image()
        #Smoothen the image
        self.img.imarray = self.img.convolve(self.gaussian_filter)

        #Computing the horizontal and vertical sobel filtering operation
        G_x = self.img.convolve(self.sobel_x)
        G_y = self.img.convolve(self.sobel_y)
        #Computing the magnitude of filtered image
        G = pow((G_x*G_x + G_y*G_y),0.5)
        G = (G>24)*G                                                        #Applying threshold value
        temp_img = self.img.convolve(self.mask)
        if (temp_img == None).any():
            return
        cimg.imarray = temp_img

        cimg.show(name = "DOG.jpg")
        (M,N) = temp_img.shape
        #detect zero crossing by checking values across 8-neighbors on a 3x3 grid
        temp = np.zeros((M+2,N+2))
        temp[1:-1,1:-1] = temp_img
        img = np.zeros((M,N))
        for i in range(1,M+1):
            for j in range(1,N+1):
                if temp[i,j]<0:
                    #Checking over 8 neighbor grid for change in polarity of the gradient
                    for x,y in (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1):
                            if temp[i+x,j+y]>0:
                                img[i-1,j-1] = 1
        cimg.imarray = img

        cimg.show(name = "DOGZC.jpg")

        cimg.imarray = np.logical_and(img,G)

        cimg.show(name = "DOGSobel.jpg")
