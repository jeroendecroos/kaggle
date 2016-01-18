# Loading Data
import csv
import time
import pprint
import sys
pre_start = time.time()
import pandas as pd 
from skimage.io import imread
import numpy as np
from wsgiref import headers

from sknn import ae, mlp
import sknn.backend
import cv2
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
import sklearn.neighbors
import sklearn.svm
import sklearn

import math


class Configuration():
    def __init__(self):
        self.size = 20
        if len(sys.argv) > 1:
            print sys.argv
            self.size = int(sys.argv[1])
        self.affix = ''
        if len(sys.argv) > 2:
            self.affix = sys.argv[2]
        self.grey_scale = True
        self.IMAGE_SIZE = self.size * self.size # 20 x 20 pixels
        if not self.grey_scale:
            self.IMAGE_SIZE = self.IMAGE_SIZE * 3
        self.INPUT_PATH = "/home/jeroen/Documents/kaggle/FirstSteps/"

configuration = Configuration()
print configuration.IMAGE_SIZE

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
bin_n = 16 # Number of bins
n_groups = 5
SZ=20   #Deskew

def read_image(filename):
#    img = imread(filename, as_grey=False)    
    if configuration.grey_scale:
        img = cv2.imread(filename,0)
    else:
        img = cv2.imread(filename,1)  
#    print img.shape
#    img = hog(deskew(img))
    img = hog(img)
 #   print img.shape
#    img = np.reshape(img, (1, configuration.IMAGE_SIZE))
    #img = img/256.
    return img

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

length_group = configuration.size/n_groups
half = length_group
print half
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = tuple()
    mag_cells = tuple()
    for x in range(n_groups):
        for y in range(n_groups):
            x0 = length_group *(x)
            x1 = length_group *(x+1)
            y0 = length_group *(y)
            y1 = length_group *(y+1)
            bin = bins[x0:x1,y0:y1]
            bin_cells += (bin,)
            magie = mag[x0:x1,y0:y1]
            mag_cells += (magie,)
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

def read_data(typeData, labelsInfo, configuration, affix=''):
    x = np.zeros((labelsInfo.shape[0], bin_n*n_groups*n_groups )) #configuration.IMAGE_SIZE))
#    x = np.zeros((labelsInfo.shape[0], configuration.IMAGE_SIZE))
    for (index, idImage) in enumerate(labelsInfo["ID"]):
        nameFile = "{0}/{1}Resized{4}{5}{3}/{2}.Bmp".format(configuration.INPUT_PATH, 
                                                      typeData, idImage, 
                                                      configuration.size,
                                                      configuration.affix, 
                                                      affix)
        img = read_image(nameFile)
        x[index, :] = img
    return x


def write_test_results(model, predict_to_label_transformer, input_data_transformer=None):
    labelsInfoTest = pd.read_csv("{0}/sampleSubmission.csv".format(configuration.INPUT_PATH))
    xTest = read_data("test", labelsInfoTest, configuration)
    if input_data_transformer:
        xTest = input_data_transformer.transform(xTest)
    results = model.predict(xTest)
    results = map(predict_to_label_transformer, results)
    labelsInfoTest['Class'] = results
    labelsInfoTest.to_csv('results{}.csv'.format(configuration.size),
                          cols=['ID','Class'],
                          header= ['ID','Class'],
                          index=False,
                          )

def get_x_and_y_train(affix =''):
    labelsInfoTrain = pd.read_csv("{0}/trainLabels.csv".format(configuration.INPUT_PATH))
    #Read training matrix
    xTrain = read_data("train", labelsInfoTrain, configuration, affix)
    yTrain = map(ord, labelsInfoTrain["Class"])
    yTrain = np.array(yTrain)
    return xTrain, yTrain

# Importing main functions



# Running LOOF-CV with 1NN sequentially
def simple_knn():
    start = time.time()
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(xTrain, yTrain)
#    model.fit(xTrain, yTrain)
#    cv_scorer = cross_val_score(model, xTrain, yTrain, cv=2, scoring="accuracy")
 #   cvAccuracy = np.mean(cv_scorer)
#    print "The 2-CV accuracy of 1NN", cvAccuracy
    print time.time() - start, "seconds elapsed"

def encode_x_y_train(xTrain):
    myae = ae.AutoEncoder(
            layers=[
#                ae.Layer("Tanh", units=100),
#                ae.Layer("Tanh", units=100),
                ae.Layer("Sigmoid", units=100),
                ae.Layer("Sigmoid", units=100)],
            learning_rate=0.2,
            n_iter=2)
    myae.fit(xTrain)
    return myae

def encode_x_y_train_opencv(xTrain):
   return None
 
 

#simple_knn()
#exit()
test_number = 1000
xTrain, yTrain = get_x_and_y_train()
myae = encode_x_y_train_opencv(xTrain)
if myae:
    xTrain = myae.transform(xTrain)
xTrainTest =xTrain[:test_number]
xTrain = xTrain[test_number:]
yTrainTest = yTrain[:test_number]
yTrain = yTrain[test_number:]
training_sets =(xTrain,)
training_labels = ['BilateralFiltering', 'erode', 'dilate','MorphologicalGradient']
for training_label in training_labels:
    xTrain2, yTrain2 = get_x_and_y_train(training_label)
    xTrain2 = xTrain2[test_number:]
    training_sets += (xTrain2,)
xTrain = np.concatenate(training_sets)
yTrain = np.concatenate((yTrain,) * (len(training_labels)+1))

#print xTrain[2][:10]
#print xTrain[3][:10]
#print len(xTrain[2])
myae = encode_x_y_train_opencv(xTrain)
if myae:
    xTrain = myae.transform(xTrain)
#print xTrain[2][:50]
#print xTrain[3][:50]
#exit()
#print len(xTrain[2])
#exit()
# Tuning the value for k
model = KNeighborsClassifier(n_neighbors=1)
tuned_parameters = [{"n_neighbors":list(range(1,5))}]
model = ensemble.ExtraTreesClassifier(n_estimators=10 )#, criterion="entropy")
tuned_parameters = [{"n_estimators":list(range(30,50,10)),
                     "criterion":["entropy","gini"]                   
                     }]
#tuned_parameters = [{}]
#xTrain = xTrain
#yTrain = yTrain
#print yTrain
start = time.time()
print start-pre_start
#model = sklearn.svm.LinearSVC()
#model.fit(xTrain, yTrain)
#tuned_parameters = [{"loss":['hinge','squared_hinge']
 #                    }]
#tuned_parameters = [{"kernel":['linear']}]#,'poly','rbf','sigmoid','precomputed']}]
print 'start searching'
clf = GridSearchCV( model, tuned_parameters, cv=5, scoring="accuracy")
clf.fit(xTrain, yTrain)
yTrainTestPredict = clf.predict(xTrainTest)
score = sklearn.metrics.accuracy_score(yTrainTest, yTrainTestPredict)
print 'score', score , ' +- ' , 1/math.sqrt(test_number), 1.96*math.sqrt(score*(1-score)/test_number)

write_test_results(clf, chr, myae)
pprint.pprint( clf.grid_scores_ )
print time.time() - start, "seconds elapsed"

