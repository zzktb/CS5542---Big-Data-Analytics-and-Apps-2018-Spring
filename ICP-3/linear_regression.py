from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext

# Load and parse data
def parsePoint(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return LabeledPoint(values[0], values[1:])

sc = SparkContext("local")
data = sc.textFile('data\\lpsa.data')
parsedData = data.map(parsePoint)
train_data, test_data = parsedData.randomSplit([0.7, 0.3])

# Build model
lr = LinearRegressionWithSGD.train(train_data, iterations=100, step=20.0)

# predict train
trainPreds = train_data.map(lambda p: (p.label, lr.predict(p.features)))
train_MSE = trainPreds.map(lambda p: (p[0] - p[1])**2).reduce(lambda x, y: x + y) / trainPreds.count()
print("Train data MSE = " + str(train_MSE))

# predict test
testPreds = test_data.map(lambda p: (p.label, lr.predict(p.features)))
test_MSE = testPreds.map(lambda p: (p[0] - p[1])**2).reduce(lambda x, y: x + y) / testPreds.count()
print("Test data MSE = " + str(test_MSE))

sc.stop()
