from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt

sc = SparkContext("local")

# load and parse data
data = sc.textFile("D:\\umkc\\2018Spring\\Big_data_analytics\\lecture_05\\3D_spatial_network.txt")
parsedData = data.map(lambda line: array([float(x) for x in line.replace(',', ' ').split(' ')]))

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))


# Build the model
n_cluster = [3, 4]
for i in range(2):
    clusters = KMeans.train(parsedData, n_cluster[i], maxIterations=30, initializationMode='random')
    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("n_cluster: "+str(n_cluster[i])+"    WSSSE: " + str(WSSSE))

sc.stop()