#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = SVC(kernel="rbf", C = 10000)

#features_train = features_train[:int(len(features_train)*0.01)]
#labels_train = labels_train[:int(len(labels_train)*0.01)]

print "Lenght: " + str(len(labels_train))

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t1, 3), "s"

#########################################################
### your code goes here ###

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print "Accuracy " + str(acc)

chris = 0

print "Pred"
print pred

for i in pred:
    if i == 1:
        chris = chris + 1

print "Chris"
print chris

#########################################################


