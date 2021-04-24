import glob
from PIL import Image
import FeatureExtraction
import os
import numpy as np
import cv2
import pandas as pd
import pickle
from joblib import dump, load
import pandas as pd

outpath = '/home/saipreethamsata/Desktop/ISM_Project/hairRemoved'

counter=0
X=[]


def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

def number_characters(x):
    return(x[30:38])

# Run the above function and store its results in a variable.
full_file_paths = get_filepaths('/home/saipreethamsata/Desktop/ISM_Project/ISIC_2019_Training_Input')
images = glob.glob("ISIC_2019_Training_Input/*.jpg")
counter=0
images_List=[]
dataset = pd.read_csv('ISIC_2019_Training_GroundTruth.csv')
y = dataset.iloc[:, [1,2,3,4,5,6,7,8,9]].values
for image in images:
    with open(image, 'rb') as file:
        images_List.append(image)

print(images_List)
features=[]
dirFiles = images_List
counter=0
newDir=sorted(dirFiles, key = number_characters)
for image in newDir:
    with open(image, 'rb') as file:
        images_List.append(image)
        featureExtractor=FeatureExtraction.FeatureExtraction(image)
        Image=featureExtractor.getImage()
        #Image1=featureExtractor.removeHair()
        Image1=featureExtractor.segmentation1(Image)
        harlickFeatures=featureExtractor.harlickFeatures(Image1)
        hueMoments,moments=featureExtractor.momentsAndHuemoments(Image1)
        momentsArray=[]
        momentsLabel=[]
        for i in moments.values():
            momentsArray.append(i)
        for i in moments.keys():
            momentsLabel.append(i)
        moments1=np.array(momentsArray)
        meanStandardDeviation=featureExtractor.meanStandardDeviation(Image1)
        pca_features=featureExtractor.pca_features(Image1)
        print('haralickFeatures',harlickFeatures)
        print('hueMoments',np.reshape(hueMoments,[7]))
        print('moments',moments1)
        print('meanStandardDeviation',(np.reshape(meanStandardDeviation,[6])))
        print('pca_features',(pca_features))
        totalFeatures=np.concatenate((harlickFeatures,np.reshape(hueMoments,[7]),moments1,np.reshape(meanStandardDeviation,[6]),pca_features),axis=0)
        features.append(totalFeatures)

        counter=counter+1
        print(counter)
        if counter==100:
            break

harLickHueMomentsHeader=['harlickFeature1','harlickFeature2','harlickFeature3','harlickFeature4','harlickFeature5',
'harlickFeature6','harlickFeature7','harlickFeature8','harlickFeature9','harlickFeature10','harlickFeature11','harlickFeature12','harlickFeature13',
'HueMoment1','HueMoment2','HueMoment3','HueMoment4','HueMoment5','HueMoment6','HueMoment7']
meanStandardHeader=['red Mean','red StandardDeviation','green Mean','green StandardDeviation','blue Mean','blue StandardDeviation']
pcaHeader=['PCA Feature1','PCA Feature2','PCA Feature3','PCA Feature4','PCA Feature5','PCA Feature6','PCA Feature7','PCA Feature8','PCA Feature9','PCA Feature10','PCA Feature11',
'PCA Feature12','PCA Feature13','PCA Feature14','PCA Feature15','PCA Feature16','PCA Feature17','PCA Feature18','PCA Feature19','PCA Feature20']
header=harLickHueMomentsHeader+momentsLabel+meanStandardHeader+pcaHeader


pd.DataFrame(features).to_csv("feature1.csv",header=header)


















'''
Image=cv2.imread('/home/saipreethamsata/Desktop/ISM_Project/ISIC_0026210.jpg')
featureExtractor=FeatureExtraction.FeatureExtraction('/home/saipreethamsata/Desktop/ISM_Project/ISIC_0026786.jpg')
Image=featureExtractor.getImage()
segmentedImage1=featureExtractor.segmentation(Image)
#cv2.imshow('segmentedImage1',segmentedImage1)
#cv2.waitKey()
segmentedImage2=featureExtractor.segmentation1(Image)
#cv2.imshow('segmentedImage2',segmentedImage2)
#cv2.waitKey()
kmeanSegmentation=featureExtractor.kmeansSegmentation(Image)
#cv2.imshow('kmeanSegmentation',kmeanSegmentation)
#cv2.waitKey()

harlickFeatures=featureExtractor.harlickFeatures(segmentedImage2)
print(harlickFeatures)
hueMoments,moments=featureExtractor.momentsAndHuemoments(segmentedImage2)
momentsArray=[]
for i in moments.values():
    momentsArray.append(i)

moments=np.array(momentsArray)

meanStandardDeviation=featureExtractor.meanStandardDeviation(segmentedImage2)
print(meanStandardDeviation)
pca_features=featureExtractor.pca_features(segmentedImage2)
print(np.shape(pca_features))

#toCSV
print(np.shape(harlickFeatures))
print(np.shape(np.reshape(hueMoments,[7])))
print(np.shape(moments))
print(np.shape(np.reshape(meanStandardDeviation,[6])))
print(np.shape(pca_features))

totalFeatures=np.concatenate((harlickFeatures,np.reshape(hueMoments,[7]),moments,np.reshape(meanStandardDeviation,[6]),pca_features),axis=0)
print(totalFeatures)
'''
