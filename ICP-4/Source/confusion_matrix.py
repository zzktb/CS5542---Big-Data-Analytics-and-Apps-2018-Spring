'''
compute confusion matrix given classification results.
---created by Z.Zhang 2/8/2018
'''

from pandas_ml import ConfusionMatrix

y_true = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0]
y_pred = [0, 1, 0, 1, 1, 0, 0, 1, 0, 0]

conf = ConfusionMatrix(y_true, y_pred)
acc = conf.ACC

misRate = (conf.FP + conf.FN) / 10
truePos = (conf.TP / (conf.TP + conf.FN))
falsePos = (conf.FP / (conf.FP + conf.FN))
spec = (conf.TN / (conf.FP + conf.TN))
prec = (conf.TP / (conf.TP + conf.FP))
prev = ((conf.TP + conf.TN) / 10)

print('Confusion matrix: \n%s' % conf + '\n')
print('acc: %.4f' % acc)
print('misRate: %.4f' % misRate)
print('truePos: %.4f' % truePos)
print('spec: %.4f' % spec)
print('prec: %.4f' % prec)
print('prev: %.4f' % prev)