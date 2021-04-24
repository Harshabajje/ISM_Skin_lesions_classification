import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
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





def writeMetaData(image,age,gender,position):
    Image=cv2.imread(image)
    font=cv2.FONT_HERSHEY_SIMPLEX
    width,height=Image.shape[:2]
    cv2.putText(Image,text=age,org=(10,(width-100)),fontFace=font,fontScale=1,color=(255,0,0),thickness=1,lineType=cv2.LINE_AA)
    cv2.putText(Image,text=position,org=(10,width-70),fontFace=font,fontScale=1,color=(0,255,0),thickness=1,lineType=cv2.LINE_AA)
    cv2.putText(Image,text=gender,org=(10,width-40),fontFace=font,fontScale=1,color=(0,0,255),thickness=1,lineType=cv2.LINE_AA)
    return Image


def show_image(image,path):
    fig=plt.figure(figsize=(5,8))
    ax=fig.add_subplot(111)
    ax.imshow(image)
    plt.imshow(image)
    plt.show()
    plt.savefig(path)

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
#writeMetaData('ISIC_0024592.jpg','55','anterior torso','female')

metaData=pd.read_csv('ISIC_2019_Training_Metadata.csv')

metaData=np.array(metaData)
label=metaData[:,0]
age=metaData[:,1]
lesion_position=metaData[:,2]
gender=metaData[:,4]
totalElements=age.shape[0]
full_file_paths = get_filepaths('/home/saipreethamsata/Desktop/ISM_Project/ISIC_2019_Training_Input')
images = glob.glob("ISIC_2019_Training_Input/*.jpg")
counter=0
images_List=[]
#dataset = pd.read_csv('ISIC_2019_Training_GroundTruth.csv')
#y = dataset.iloc[:, [1,2,3,4,5,6,7,8,9]].values
for image in images:
    with open(image, 'rb') as file:
        images_List.append(image)

print(images_List)
features=[]
dirFiles = images_List
counter=0
path1 = '/home/saipreethamsata/Desktop/ISM_Project/metaDataLabeledImages2/'
path2 = '/home/saipreethamsata/Desktop/ISM_Project/metaDataLabeledImages3/'
path3 = '/home/saipreethamsata/Desktop/ISM_Project/metaDataLabeledImages4/'
path4 = '/home/saipreethamsata/Desktop/ISM_Project/metaDataLabeledImages5/'
newDir=sorted(dirFiles, key = number_characters)
for i,image in enumerate(newDir):
    with open(image, 'rb') as file:
        print(image)
        print(label[i])
        print(i)
        print(age[i])
        print(gender[i])
        print(lesion_position[i])
        Image=writeMetaData(image,str(age[i]),str(gender[i]),str(lesion_position[i]))
        counter=counter+1
        print(counter)
        if counter>=0 and counter <6000:
            cv2.imwrite(os.path.join(path1,image[25:]),Image)
        elif counter>=6000 and counter <12000:
            cv2.imwrite(os.path.join(path2,image[25:]),Image)
        elif counter>=12000 and counter <18000:
            cv2.imwrite(os.path.join(path3,image[25:]),Image)
        elif counter>=18000:
            cv2.imwrite(os.path.join(path4,image[25:]),Image)
