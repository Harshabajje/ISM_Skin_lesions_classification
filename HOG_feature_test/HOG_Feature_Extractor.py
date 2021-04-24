import glob
from PIL import Image
import FeatureExtraction
import os
import numpy as np
import cv2
import pandas as pd
import pickle
from joblib import dump, load

#This file is exclusively Used for extracting HOF Features and storing them as dat file.The Imageloade.py is experimentation file.

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

dirFiles = images_List
newDir=sorted(dirFiles, key = number_characters)
for image in newDir:
    with open(image, 'rb') as file:
        images_List.append(image)
        featureExtractor=FeatureExtraction.FeatureExtraction(image)
        Image=featureExtractor.getImage()
        Image1=featureExtractor.segmentation1(Image)
        X.append(featureExtractor.hogFeatures(Image1,9,16,2))
        counter=counter+1
        print(np.shape(featureExtractor.hogFeatures(Image1,9,16,2)))
        print(counter)

filename_scaler = 'HOG_Features_25000_Images.dat'
pickle.dump(X, open(filename_scaler, 'wb'))

filename_scaler = 'HOG_Labels_25000_Images.dat'
pickle.dump(y, open(filename_scaler, 'wb'))
