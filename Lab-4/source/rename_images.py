'''
Rename the image names to avoid spaces
---created by Z.Zhang 3/22/2018
'''
from imutils import paths
import os
from input_fn import input_fn

dataPath = "D:\\umkc\\2018Spring\\Big_data_analytics\\deep-learning-visual-eCommerce-master\\fashion-item-dataset\\data4\\train4"
count = 0
for className in os.listdir(dataPath):
    imlist = os.listdir(os.path.join(dataPath, className))
    for i in range(len(imlist)):
        src = os.path.join(dataPath, className, imlist[i])
        dst = os.path.join(dataPath, className, str(os.listdir(dataPath).index(className))+'_IMG_'+str(count)+'.jpg')
        os.rename(src, dst)
        count += 1


