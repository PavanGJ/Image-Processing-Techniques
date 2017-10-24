import EdgeDetector as ed


dog = ed.DiffofGaussian()
dog.loadImage('./images/UBCampus.jpg')
dog.performDOG()

log = ed.LaplaceofGaussian()
log.loadImage('./images/UBCampus.jpg')
log.performLOG()
