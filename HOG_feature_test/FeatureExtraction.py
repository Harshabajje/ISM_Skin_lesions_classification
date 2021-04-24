import cv2
import numpy as np
from matplotlib import pyplot as plt
import mahotas as mt
from skimage.feature import hog
from sklearn.decomposition import IncrementalPCA


N_COMPS=20
class FeatureExtraction(object):

    def __init__(self,image):
        self.image=cv2.imread(image)

    def getImage(self):
        return self.image


    def removeHair(self):
        rgbImage=cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
        grayImage=cv2.cvtColor(self.image,cv2.COLOR_RGB2GRAY)
        blurImage=cv2.GaussianBlur(grayImage,(7,7),0)
        thresholdImage=cv2.adaptiveThreshold(blurImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,0)
        kernel=np.ones((7,7),np.uint8)
        closeImage=cv2.morphologyEx(thresholdImage,cv2.MORPH_CLOSE,kernel)
        openImage=cv2.morphologyEx(closeImage, cv2.MORPH_OPEN, kernel)
        inpaintImage=cv2.inpaint(self.image,openImage,50,cv2.INPAINT_TELEA)
        return inpaintImage


    def segmentation(self,image):
        rgbImage=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        grayImage=cv2.cvtColor(rgbImage,cv2.COLOR_RGB2GRAY)
        blurImage=cv2.GaussianBlur(grayImage,(3,3),0)
        ret2,thresholdImage = cv2.threshold(blurImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel=np.ones((25,25),np.uint8)
        closeImage=cv2.morphologyEx(thresholdImage,cv2.MORPH_CLOSE,kernel)
        openImage=cv2.morphologyEx(closeImage,cv2.MORPH_OPEN,kernel)
        kernel1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
        openImage1=cv2.morphologyEx(openImage, cv2.MORPH_OPEN, kernel1)
        mask=cv2.bitwise_not(openImage1)
        finalImage=cv2.bitwise_and(image,image,mask=mask)
        return finalImage

    def segmentation1(self,image):
        rgbImage=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        m,n,c=np.shape(rgbImage)
        grayImage=cv2.cvtColor(rgbImage,cv2.COLOR_RGB2GRAY)
        blurImage=cv2.GaussianBlur(grayImage,(3,3),0)
        ret2,thresholdImage = cv2.threshold(blurImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(m,n))
        mask=mask.T
        for i in range(m):
            for j in range(n):
                if mask[i,j]==1:
                     mask[i,j]=255
                else:
                     mask[i,j]=0
        finalImage=cv2.bitwise_and(image,image,mask=mask)
        return finalImage

    def kmeansSegmentation(self,image):
        rgbImage=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        vectorized = rgbImage.reshape((-1,3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 15
        attempts=10
        ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((rgbImage.shape))
        print(np.shape(result_image))
        return result_image


    def pca_features(self,image):
        pca = IncrementalPCA(n_components=N_COMPS)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #X=gray.flatten()
        X_pca = pca.fit_transform(gray)
        variance_Ratio=pca.explained_variance_ratio_
        return variance_Ratio
        
    def fourierTransform(self,image):
        rgbImage=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        grayImage=cv2.cvtColor(rgbImage,cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(grayImage)
        fshift = np.fft.fftshift(grayImage)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        return magnitude_spectrum

    def complexityFeatureSet(self,Image):
        complexityFeatures=[]
        mean,standardDeviation=cv2.meanStdDev(Image)
        complexityFeatures=[mean,standardDeviation]
        return complexityFeatures

    def harlickFeatures(self,image):
        textures=mt.features.haralick(image)
        ht_mean=textures.mean(axis=0)
        return ht_mean

    def hogFeatures(self,image,num_orientation,pix_per_cell,cell_per_block):
        features = hog(image, orientations=num_orientation,
					   pixels_per_cell=(pix_per_cell, pix_per_cell),
					   cells_per_block=(cell_per_block, cell_per_block),
					   transform_sqrt=True,
					   feature_vector=True)

        return features

    def colorHistograms(self,image):
        hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return hist

    def hueMoments(self,image):
        rgbImage=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        grayImage=cv2.cvtColor(rgbImage,cv2.COLOR_RGB2GRAY)
        blurImage=cv2.GaussianBlur(grayImage,(3,3),0)
        ret2,thresholdImage = cv2.threshold(blurImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        moments = cv2.moments(thresholdImage)
        huMoments = cv2.HuMoments(moments)
        return huMoments

