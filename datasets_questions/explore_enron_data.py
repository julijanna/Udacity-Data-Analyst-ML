#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "Data length"
print len(enron_data)

print "SKILLING JEFFREY K"
print len(enron_data["SKILLING JEFFREY K"])

poi_count = 0

for person in enron_data:
    if enron_data[person]["poi"] == 1:
        poi_count += 1
        print person

print "POIs"
print poi_count

print "POI list"
poi_list = 0

crs = open("../final_project/poi_names.txt", "r")
for row in crs:
    if row[:1] == "(":
        poi_list += 1

print poi_list

print "James Prentice stock"
print enron_data["PRENTICE JAMES"]["total_stock_value"]

print "WESLEY COLWELL messages"
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

print "SKILLING JEFFREY K"
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print "SKILLING JEFFREY K payments"
print enron_data["SKILLING JEFFREY K"]["total_payments"]

print "FASTOW ANDREW payments"
print enron_data["FASTOW ANDREW S"]["total_payments"]

print "LAY KENNETH payments"
print enron_data["LAY KENNETH L"]["total_payments"]

salary_count = 0
email_count = 0
total_payments = 0
poi_nan_payments = 0
for person in enron_data:
    if enron_data[person]["salary"] != "NaN":
        salary_count += 1
    if enron_data[person]["email_address"] != "NaN":
        email_count += 1
    if enron_data[person]["total_payments"] == "NaN":
        total_payments += 1
    if enron_data[person]["total_payments"] == "NaN" and enron_data[person]["poi"] == 1:
        poi_nan_payments += 1

print "Salary count"
print salary_count

print "Email count"
print email_count

print "Total payment NaNs"
print total_payments

print "POI payment NaNs"
print poi_nan_payments