'''

---created by Z.Zhang 3/22/2018
'''
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from imutils import paths
import numpy as np
import imutils
import cv2
import os

dataPath = "D:\\umkc\\2018Spring\\Big_data_analytics\\deep-learning-visual-eCommerce-master\\fashion-item-dataset\\data4"

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()

# initialize the data matrix and labels list
data_train = []
labels_train = []
data_test = []
labels_test = []

# loop over the input images
for stage in ['train4', 'test4']:
    classList = os.listdir(os.path.join(dataPath, stage))
    for className in classList:
        imagePaths = os.path.join(dataPath, stage, className)
        for imageName in list(paths.list_images(imagePaths)):
            imagePath = os.path.join(dataPath, stage, className, imageName)
            image = cv2.imread(imagePath)
            label = className
            hist = extract_color_histogram(image)
            if stage is 'train4':
                data_train.append(hist)
                labels_train.append(label)
            elif stage is 'test4':
                data_test.append(hist)
                labels_test.append(label)

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels_train = le.fit_transform(labels_train)
labels_test = le.fit_transform(labels_test)

# train the linear regression clasifier
print("[INFO] constructing SGD classifier...")
model = SGDClassifier(loss="log", random_state=967, n_iter=100)
model.fit(data_train, labels_train)

# evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = model.predict(data_test)
print(classification_report(labels_test, predictions,
                            target_names=le.classes_))