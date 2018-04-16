import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import svm
import seaborn
import pandas as pd
from matplotlib import pyplot as plt

#Load target
type_file = "type.csv"
cuisine = pd.read_csv(type_file)
target = {}
i = 0
for cuisineName in cuisine.Cuisine:
    target[cuisineName.lower()] = cuisine.CuisineNumber[i]
    i += 1

#Load features
categories = pd.read_csv(type_file)
features_category = {}
i = 0
for category in categories.TypeName:
    features_category[category.lower()] = []
    features_category[category.lower()].append(categories.TypeNumber[i])
    features_category[category.lower()].append(categories.Price[i])
    i += 1
feature = []

for category in categories.TypeName:
    feature.append(features_category[category.lower()])

features = np.array(feature)
X, y = features, target.keys()
model = KMeans(n_clusters=12)
model.fit(X)
#print model.cluster_centers_
    dictLabels = {0:"Head", 1:"Trail", 2:"Tire", 3:"Panel"}


#Enter price value
predictCuisine = [7500]
print "KMeans predicted cuisine as '%s'." %dictLabels[model.predict(predictCuisine)[0]]

print "Accuracy Score: %d" %model.score(X)

distortions = []
for i in range(1, 13):
    km = KMeans(n_clusters = i,
               init='k-means++',
               n_init=10,
               max_iter=300,
               random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1,13), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

