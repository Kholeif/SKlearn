#Import Libraries
from sklearn.datasets import load_digits
#----------------------------------------------------

#load digits data

DigitsData = load_digits()

#X Data
X = DigitsData.data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)

#y Data
y = DigitsData.target
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)

import matplotlib.pyplot as plt
plt.gray()

for g in range(5):
    print('Images of Number : ' , g)
    plt.matshow(DigitsData.images[g])
    print('------------------------------')
    plt.show()
    
import cv2
for g in range(10):
    cv2.imshow("lalloo",cv2.resize(DigitsData.images[g],(32,32)))
    print(y[g])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
