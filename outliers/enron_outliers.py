#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL", 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

new_data = sorted(data,key=lambda l:l[0], reverse=False)


for person in data_dict:
    if data_dict[person]['salary'] == new_data[len(data)-1][0]:
        print person
    if data_dict[person]['salary'] == new_data[len(data) - 2][0]:
        print person
    if data_dict[person]['salary'] == new_data[len(data) - 3][0]:
        print person
    if data_dict[person]['salary'] == new_data[len(data) - 4][0]:
        print person

### your code below



for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()