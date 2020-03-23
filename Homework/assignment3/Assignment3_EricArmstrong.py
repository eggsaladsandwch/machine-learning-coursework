"""
@author: Eric Armstrong
"""

# got lots of help from Zachary Combs on this, some of this code is his

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from pyimagesearch.preprocessing import SimplePreprocessor
#from pyimagesearch.datasets import SimpleDatasetLoader
#from imutils import paths
import numpy as np
#from scipy.misc import imread, imresize
import cv2
import os 
  
print("[INFO] loading images...")

DataPath = "./Data/KNN/animals/"
folder_list = os.listdir(DataPath)

data = []
labels = []

for foldername in folder_list:
    imagePath = DataPath + foldername
    
    if imagePath == './Data/KNN/animals/.DS_Store':
            continue
    
    ImageList = os.listdir(imagePath)
    for ImageName in ImageList:
        imageFullPath = DataPath + foldername + "/" + ImageName
        image = cv2.imread(imageFullPath)
        image = cv2.resize(image, (32, 32), interpolation = cv2.INTER_CUBIC)
        data.append(image)
        labels.append(foldername)
        
    print("Subfolder Done")
    
FinalData = np.array(data)
FinalLabel = np.array(labels)

FinalData = FinalData.reshape((FinalData.shape[0], 3072))

le = LabelEncoder()
mylabels = le.fit_transform(FinalLabel)

#Testing set
(trainX, testX, trainY, testY) = train_test_split(FinalData, mylabels,
    test_size = 0.20, random_state = 42)

#Validation Set
(trainXv, valX, trainYv, valY) = train_test_split(trainX, trainY,
    test_size = 0.10, random_state = 42)

print("[INFO] evaluating k-NN classifier...")

model = KNeighborsClassifier(n_neighbors = 3, p = 1)
model.fit(trainXv, trainYv)
predY = model.predict(testX)
print(classification_report(testY, predY, target_names = le.classes_))