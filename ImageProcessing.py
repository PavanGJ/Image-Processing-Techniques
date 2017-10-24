import numpy as np
from matplotlib import pyplot as plot
from scipy.ndimage import imread

class Image() :
    def __init__(self) :
        self.image = None
        self.type = None
        self.dimensions = []
    def open(self,path,mode = 'L') :
        try :
            self.image = imread(path, mode = mode)
        except :
            print("Error! Could not read the image from the path specified: %s"%path)
            return
        
        try :
            self.image = np.asarray(self.image, dtype = float)
            self.dimensions = self.image.shape
            self.type = path.split(".")[-1]
        except :
            print("Internal Error! Image file not supported")
    def set(self, image) :
        image = np.asarray(image, dtype = float)
        if len(image.shape) == 2 :
            self.image = image
            try :
                self.dimensions = self.image.shape
            except :
                print("Internal Error! Image file not supported")
        else :
            print("Assignment Error. Given input is not an image")
    def show(self, mode='Greys_r', name=None) :
        try :
            plot.imshow(self.image,cmap=mode)
        except :
            print("Image Could not be displayed")
            return
        if not name is None :
            plot.imsave(name,self.image,cmap=mode)
        plot.show()

class FourierTransform() :
    def __init__(self) :
        self.f = None
        self.F = None
        self.magnitude = None
        self.phase = None
        self.M = None
        self.N = None
        self.image = None
    def setImage(self, image) :
        image = np.asarray(image, dtype = float)
        if len(image.shape) == 2 :
            self.image = image
            try :
                self.dimensions = self.image.shape
            except :
                print("Internal Error! Image file not supported")
        else :
            print("Assignment Error. Given input is not an image")
    def forwardTransform(self) :
        try :
            M = self.image.shape[0]
            N = self.image.shape[1]
        except :
            print("Internal Error! Could not decompose the image shape")
            return
        x = np.arange(M, dtype = float)
        y = np.arange(N, dtype = float)
        u = x.reshape((M,1))
        v = y.reshape((N,1))
        exp_1 = pow(np.e, -2j*np.pi*u*x/M)
        exp_2 = pow(np.e, -2j*np.pi*v*y/N)
        self.F = np.dot(exp_2, np.dot(exp_1,self.image).transpose())/(M*N)
        return self.F

    def inverseTransform(self) :
        try :
            M = self.F.shape[0]
            N = self.F.shape[1]
        except :
            print("Internal Error! Could not decompose the image shape")
            return
        x = np.arange(M, dtype = float)
        y = np.arange(N, dtype = float)
        u = x.reshape((M,1))
        v = y.reshape((N,1))
        exp_1 = pow(np.e, 2j*np.pi*u*x/M)
        exp_2 = pow(np.e, 2j*np.pi*v*y/N)
        self.f = np.dot(exp_2, np.dot(exp_1,self.F).transpose())
        return self.f
    def shift(self, image) :
        try :
            M = image.shape[0]
            N = image.shape[1]
        except :
            print("Internal Error! Could not decompose the image shape")
            return
        m = M/2
        n = N/2
        temp = np.zeros((M,N))
        temp[-m:,-n:] = np.abs(np.copy(image[:m,:n]))
        temp[-m:,:-n] = np.abs(np.copy(image[:m,n:]))
        temp[:-m,-n:] = np.abs(np.copy(image[m:,:n]))
        temp[:-m,:-n] = np.abs(np.copy(image[m:,n:]))
        return temp
    def error(self) :
        E = (self.image - self.f)**2
        M = E.shape[0]
        N = E.shape[1]
        I = np.ones((1,N))
        J = np.ones((M,1))
        print("Error: %s"% np.abs(np.dot(np.dot(I,E.transpose()),J)))

class LaplacianPyramid() :
    def __init__(self) :
        self.gaussian_filter = np.array([[1/16.,1/8.,1/16.],[1/8.,1/4.,1/8.],[1/16.,1/8.,1/16.]])
    def setImage(self, image) :
        self.original = image      
        self.M = image.shape[0]
        self.N = image.shape[1]
    def subsample(self) : 
        self.subSampled = self.original[::2,::2]       
        return self.subSampled
    def smoothen(self):
        paddedImage = np.ones((self.M+2,self.N+2))*128
        paddedImage[1:-1,1:-1] = self.original

        for m in range(self.M) :
            for n in range(self.N) :
                paddedImage[m+1,n+1] = sum(sum(paddedImage[m:m+3,n:n+3]*self.gaussian_filter))

        self.smoothened = paddedImage[1:-1,1:-1]
        return self.smoothened
    def highPassFilter(self) :
        self.highPass = self.original-self.upsample()
        return self.highPass
    def upsample(self, image=None) :
        if image is None :
            image = self.subSampled
        m = image.shape[0]
        n = image.shape[1]
        upsample = np.ones((2*m,2*n))
        upsample[::2,::2] = image
        upsample[1::2,1::2] = image
        upsample[::2,1::2] = image
        upsample[1::2,::2] = image
        return upsample
    def reconstruct(self,image) :
        self.reconstructed = self.highPass + image
        return self.highPass + image
    def error(self,original,image) :
        E = (original - image)**2
        M = E.shape[0]
        N = E.shape[1]
        I = np.ones((1,N))
        J = np.ones((M,1))
        print("Error: %s"% np.abs(np.dot(np.dot(I,E.transpose()),J)))


class ImageProcessing() :
    def __init__(self) :
        self.image = Image()
        self.fourierTransform = FourierTransform()
        self.laplacianLevels = []
    def readImage(self,path) :
        self.image.open(path)
    def showImage(self) :
        self.image.show()
    def computeFourierTransforms(self) :
        self.fourierTransform.setImage(self.image.image)
        fimg = Image()
        fimg.set(np.log(np.abs(self.fourierTransform.shift(self.fourierTransform.forwardTransform()))**2))
        fimg.show(name='images/forwardTransform.jpg')
        infimg = Image()
        infimg.set(np.abs(self.fourierTransform.inverseTransform()))
        infimg.show(name='images/inverseTransform.jpg')
        self.fourierTransform.error()
    def computeLaplacianPyramids(self) :
        img = self.image
        for i in range(5) :
            lvlImg = LaplacianPyramid()
            lvlImg.setImage(img.image)
            smoothenedImage = Image()
            smoothenedImage.set(lvlImg.smoothen())
            subSampledImage = Image()
            subSampledImage.set(lvlImg.subsample())
            highPassImage = Image()
            highPassImage.set(lvlImg.highPassFilter())
            highPassImage.show(name='images/HighPass Lvl'+str(i+1)+'.jpg')
            img = subSampledImage
            self.laplacianLevels.append(lvlImg)

        image = Image()
        image.set(self.laplacianLevels[4].original)
        image.show(name='images/Reconstructed Lvl'+str(4)+'.jpg')
        for j in reversed(range(1,5)) :
            reconstructed = self.laplacianLevels[j-1].reconstruct(self.laplacianLevels[j-1].upsample())
            image = Image()
            image.set(reconstructed)
            image.show(name='images/Reconstructed Lvl'+str(j-1)+'.jpg')
        self.laplacianLevels[0].error(self.image.image,image.image)
