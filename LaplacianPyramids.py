import ImageProcessing as ip

imageProcessing = ip.ImageProcessing()
imageProcessing.readImage('images/image.jpg')
imageProcessing.showImage()
imageProcessing.computeLaplacianPyramids()
