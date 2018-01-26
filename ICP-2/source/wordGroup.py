from pyspark import SparkConf, SparkContext

sc = SparkContext("local")

g_words = sc.textFile("input.txt").flatMap(lambda x: x.split(" ")).groupBy(lambda x: x[0])

# print(g_words.collect())

output = [(w[0], [i for i in w[1]]) for w in g_words.collect()]
file = open("output.txt", 'w')
file.write("\n".join(str(x) for x in output))
file.close()

sc.stop()
