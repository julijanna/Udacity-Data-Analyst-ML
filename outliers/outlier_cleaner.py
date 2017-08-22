#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    tuples = []

    i = 0

    while i < len(predictions):
        tuples.append((ages[i][0], net_worths[i][0], abs(predictions[i][0] - net_worths[i][0])))
        print tuples[i]
        i += 1

    sorted_tuples = sorted(tuples, key=lambda tup: tup[2])

    print "sorted"
    print sorted_tuples

    cleaned_data = sorted_tuples[:81]

    ### your code goes here

    
    return cleaned_data

