#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.feature_selection import SelectKBest

def isNaN(num):
    return num != num

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#financial features: ['salary', 'deferral_payments', 'total_payments',
# 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
# 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
# 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

#email features: ['to_messages', 'email_address', 'from_poi_to_this_person',
#  'from_this_person_to_poi', 'shared_receipt_with_poi']

#POI label: ['poi'] (boolean, represented as integer)

# features list is under my features
# You will need to use more
# features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#checking number of data points
# n_data_points = 0
#
# for person in data_dict:
#     n_data_points += 1
#
# print "Data points: ", n_data_points

### Task 2: Remove outliers
import matplotlib.pyplot as plt

person_no_nans = {}

for person, feature in data_dict.iteritems():
    n_info = 0
    for key, value in feature.iteritems():
        if value != 'NaN':
            n_info += 1
    feature['number_info'] = n_info
    person_no_nans[person] = n_info

import operator

sorted_person_no_nans = sorted(person_no_nans.items(),
                              key=operator.itemgetter(1))

# printing sorted dictionary
# for person, nans in sorted_person_no_nans:
#     print person, nans

# one person has only nans, so i'll remove him
data_dict.pop("LOCKHART EUGENE E", 0)

# there is also "travel agency in the park" -> it doesn't seem as a real person
# I'll remove it
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

# TOTAL also doesn't seem like a real person
data_dict.pop("TOTAL", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# new feature = number of information we have about this person
# (maybe pois had more different payments and less NaNs?)

for person, feature in data_dict.iteritems():
    n_info = 0
    for key, value in feature.iteritems():
        if value != 'NaN':
            n_info += 1
    feature['number_info'] = n_info

##fraction from poi
for person, feature in data_dict.iteritems():
    ratio = 0
    feature['ratio_from_poi'] = \
        float(feature['from_poi_to_this_person'])/float(feature['to_messages'])
    if isNaN(feature['ratio_from_poi']):
        feature['ratio_from_poi'] = 0

##fraction to poi
for person, feature in data_dict.iteritems():
    ratio = 0
    feature['ratio_to_poi'] =\
        float(feature['from_this_person_to_poi'])/float(feature[
                                                            'from_messages'])
    if isNaN(feature['ratio_to_poi']):
        feature['ratio_to_poi'] = 0

##bonus to salary
for person, feature in data_dict.iteritems():
    ratio = 0
    feature['bonus_to_salary'] = \
        float(feature['bonus']) / float(feature['salary'])
    if isNaN(feature['bonus_to_salary']):
        feature['bonus_to_salary'] = 0

my_dataset = data_dict

### Extract features and labels from dataset for local testing
features_list = ['poi', 'bonus', 'expenses', 'exercised_stock_options',
                 'other', 'shared_receipt_with_poi', 'ratio_to_poi']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split=2)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.3, random_state=42)
recall_list = []
precision_list = []
accuracy_list = []
feature_importances_list = []
features = np.array(features)
labels = np.array(labels)

for train_index, test_index in sss.split(features, labels):
    features_train, features_test = features[train_index], features[
        test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    accuracy_list.append(accuracy_score(labels_test, pred))
    recall_list.append(recall_score(labels_test, pred))
    precision_list.append(precision_score(labels_test, pred))
    #feature_importances_list.append(pipeline.steps[0][1].feature_importances_)

print "Accuracy mean: ", np.mean(accuracy_list)
print "Precision mean: ", np.mean(precision_list)
print "Recall mean: ", np.mean(recall_list)


# feature_importances_mean = np.mean(feature_importances_list, axis=0)
# print "\nMean importances: "
#
# i = 0
# while i < len(features_list) - 1:
#     print features_list[i+1], ": ", "%.3f" % feature_importances_mean[i]
#     i += 1


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)