#Enron Project

##1. Project overwiev

### 1.1 Goal

The goal of this project was to identify POIs (persons of interest), who were
 involved in Enron fraud scandal. The identification should be performed 
 using machine learning on data which were included in official Enron dataset
  and already prepared by Udacity.
 
### 1.2 Dataset

The dataset is an official Enron dataset, which can be downloaded by
everyone. It contains emails of the Enron employess (but not all of them). 
Emails are divided into folders by the name of the sender, and then divided 
in folders as there were in the email box (eg. "sent", "client__x" etc.). The
 POIs (people who actually did do any fraud)list, which is used for supervised 
 learning, was created on basis of articles which described the case and 
 guilty people. There are also financial features, which include payments to 
 employee and their stock options.
 
 The dataset includes 20 features and 148 people (data points), of which 18 are 
 indentified as POI.

### 1.3 Outliers

I had a look on NaNs in the data. There was one person, who has only NaNs
("LOCKHART EUGENE E"). I googled him, but didn't find anything related to 
Enron, so I removed him from the data.

Scrolling through names I also saw "THE TRAVEL AGENCY IN THE PARK" and 
"TOTAL", which most likely aren't real people, so I removed them, too.


# 2. Features

### 2.1 Feature selection

I decided to remove email adress, as it might work the same way as the 
signature in Susan emails - it has first name and surname and most likely these
 are not data we would use to predict something where labels are also first 
 name and surname.
 
Regarding feature selection, I was testing different methods. I used 
`SelectKMeans` for some of the algorithms, but not in all of the it enhanced 
it's performance.

I other ones, I ran algorithm multiple times and excluded features with 0 
feature importance. It was often better strategy.

After chosing features, I tried PCA for all algorithms, but here also, it 
didn't improve performance of all algorithms.

In my final algorithm, I chosen following features: `features_list = ['poi', 
'deferred_income', 'other', 'restricted_stock', 'shared_receipt_with_poi', 
'ratio_to_poi', 'number_info']`. I haven't chosen neither PCA nor 
SelectKMeans, because it showed lowed performance with either of these or both.
 
### 2.2 Feature scaling

After PCA, I decided to do scaling, as financial data is much bigger than 
number of emails sent / received. This way, all features should be treaten 
equally. I used MinMaxScaler to scale all features in range [0,1], what means
 that all features will be minimally 0 and maximally 1.
 
### 2.3 Additional feature

I noticed, that number of NaNs varies from person to person, so I created a 
new feature, which counts number of Nans. I may be, that NaN, which in this 
case is most likely 0, mean that no payments were made. So, if person has 
many features, where payments was 0, the probability, that the person was 
POI, may be lower.

I also created two features `'ratio_to_poi'` and `'ratio_from_poi'`, which give 
the fraction of all emails sent or received divided by all emails sent / 
received.

Last feature I created was `bonus_to_salary` what is fraction of bonus to 
salary.

I ended up using only `'ratio_to_poi'` and `'number_info'` in my final model.

### 2.4 Comparison

Table with algorithm performance for every features choosing and adjusting 
strategy:

| Method        | Precision           | Recall  |
| :-----------: |:-------------------:| :------:|
| all features, no scaling | 0.256938664114 | 0.3022 |
| MinMaxScaler() |  0.262724941725 | 0.3054 |
| SelectKBest(k=8) + MinMaxScaler() | 0.268406524032 | 0.2984 |
| manual selection + MinMaxScaler() | 0.305353096903 | 0.3462 |

Final features importances:

`Feature importances:  [ 0.14666049  0.1476645   0.02820934  0.07073386  0.33551516  0.27121665]`

## 3. Algorithm

I tried decision trees, random forest, SVM and AdaBoost.

### 3.1 Random Forest

Testing random forest, I used SelectKBest with different parameters. After 
it, I tried PCA, but it didn't optimize performance of the algorithm. Lastly,
 I added feature scaling with MinMaxScaler. However, the performance of this 
 algorithm was not enough.
  
### 3.2 Decision Trees
  
Then I tried decision trees. In the beginning, as in random forest, I removed
 features, which had importance 0. These are features, which I ended up using: 
 `features_list = ['poi', 'bonus', 'deferred_income', 'other', 
 'restricted_stock', 'shared_receipt_with_poi', 'ratio_to_poi', 'number_info']`.
 I was also testing best minimal samples split, and the best result was by 
 `min_samples_split=2`. Then, I tried to use PCA with different components 
 number. What I found interesting, the highest performance was achieved when
  PCA was set up to 1... but after validation it wasn't high enough. So, I 
  decided to not use PCA. I used scaling method `MinMaxScaler()`.

### 3.3 SVM

I decided for a bit different approach testing SVM. I started testing SVM with 
all possible features (apart email address) and different kernels. Best 
kernel, and only one, which had better performance than 0 in both precision 
and recall, was `kernel="poly`. Then I decided to test different C values, 
beginning with 10 and multiplyting it by 10 every time. As with kernel, there
 were only two values which performances where higher than 0 and there were 
 1000 and 10000. I decided to use better peformance `C=10000`. I also 
 tested multiple gamma values, and the best was 0.11. Then, I used 
 PCA and scaling (`MinMaxScaler()`) and tried different PCA components numbers.

```
n_components:  1 	Accuracy mean:  0.888837209302 	Precision mean:  0.193333333333 	Recall mean:  0.056
n_components:  2 	Accuracy mean:  0.886511627907 	Precision mean:  0.195 	                Recall mean:  0.054
n_components:  3 	Accuracy mean:  0.886434108527 	Precision mean:  0.211666666667 	Recall mean:  0.0593333333333
n_components:  4 	Accuracy mean:  0.886976744186 	Precision mean:  0.244166666667 	Recall mean:  0.0685
n_components:  5 	Accuracy mean:  0.887023255814 	Precision mean:  0.267633333333 	Recall mean:  0.0764
n_components:  6 	Accuracy mean:  0.886472868217 	Precision mean:  0.277166666667 	Recall mean:  0.08
n_components:  7 	Accuracy mean:  0.885880398671 	Precision mean:  0.289952380952 	Recall mean:  0.0854285714286
n_components:  8 	Accuracy mean:  0.885203488372 	Precision mean:  0.306208333333 	Recall mean:  0.09175
n_components:  9 	Accuracy mean:  0.883798449612 	Precision mean:  0.309648148148 	Recall mean:  0.0966666666667
n_components:  10 	Accuracy mean:  0.882465116279 	Precision mean:  0.313466666667 	Recall mean:  0.1022
n_components:  11 	Accuracy mean:  0.880909090909 	Precision mean:  0.314303030303 	Recall mean:  0.107090909091
n_components:  12 	Accuracy mean:  0.879709302326 	Precision mean:  0.316569444444 	Recall mean:  0.111666666667
n_components:  13 	Accuracy mean:  0.878443649374 	Precision mean:  0.318730769231 	Recall mean:  0.115538461538
n_components:  14 	Accuracy mean:  0.877209302326 	Precision mean:  0.31994047619 	        Recall mean:  0.119571428571
n_components:  15 	Accuracy mean:  0.875550387597 	Precision mean:  0.320828306878 	Recall mean:  0.124533333333
n_components:  16 	Accuracy mean:  0.873444767442 	Precision mean:  0.319477895369 	Recall mean:  0.13125
n_components:  17 	Accuracy mean:  0.870793433653 	Precision mean:  0.314810682781 	Recall mean:  0.138117647059
n_components:  18 	Accuracy mean:  0.868294573643 	Precision mean:  0.31057516095 	        Recall mean:  0.145111111111
```

### 3.4 AdaBoost

Out of my curiosity, I wanted also to check AdaBoost, as this is Katie's 
favourite algorithm. Also using PCA and scaling along with testing different 
estimators number, I didn't get better results than with decision trees.

### 3.5 Results

I decided to use decision trees because of the best performance.

|Algorithm|Precision mean|Recall mean|
|:---:|:---:|:---:|
|Random Forest| 0.348751984127 | 0.1608 |
|Decision Trees|0.305353096903|0.3462|
|SVM|0.314810682781|0.138117647059|
|AdaBoost|0.181040764791|0.136|

## 4. Parameter tuning

Parameter tuning means choosing right parameters in order to fit data the 
best way and to get best solutions.

In SVM I tuned three parameters:
- kernel
- C
- gamma

Kernel is a type of function which "divides" the points (we can visualise it 
like this on a cartesian coordinate system). The easiest, so to say, kernel 
is a linear one, which divides points using a line function `f(x) = ax + b`. 
RBF and Polynomial kernels should be used, when data cannot be divided by a 
line and this was our case. Polynomial kernel uses also a line, but with 
polynomial function, where RBF must even not necessarily be a line. For Enron
 data, polynomial kernel was the best choice.
 
C parameter considers the distance between closest points to the division 
line, made with kernel. The bigger C is, the less will be distance between 
points which are in the nearest neighborhood from the division line, which 
will be consider into algorithm. So most likely, the lower C is, less points 
will be taken into consideration, because algorithm will choose only closest 
points. On basis on only there points, algorithm will separate all points. In
 our case, `C=10000` was the best choice.
 
Gamma parameter is the parameter, which we tell how important is every point 
and how far away from it should we classify the space. The lower gamma is, 
the further away class reaches from the points.
 
## 5. Validation

Validation is a metric, which tells if the model will be also valid when used
 on general data (which is not test neither training data). Usually, already 
 too high accuracy (somewhere near 1), will tell us, that we are overfitting 
 the classifier and it will be useless with another data. In the Enron 
 example, it is a bit difficult, because we have very less data. The more 
 data we have, the more training and test cases we can generate, the more 
 sure we can be about validating the model.
 
 For validating my model I used StratifiedShuffleSplit with 100 tries. I 
 chosen test size by 0.3 and random state at 42. Then, I saved every 
 accuracy, precision and recall and took average from them. I chosen the 
 algorithm, which had the highest average.
 
 ## 6. Performance metrics
 
 To measure my performance I used precision and recall. I had an eye on 
 accuracy, but it wasn't important metric for me.
 
 Precision tells us, how many people we identified correctly. In this metric
  are also people, who may not really be POIs. If it is high it would be good
   if a list of these people were sent to court for further identification. 
   If precision equals one, we are sure that we haven't skipped any POIs.
   
 Recall tells us also how many people we identified correctly. In opposition 
 to performance, recall shows, how many people are not identified incorrectly
 . It means, that if recall equals 1, we are sure, that all identified people
  can go straight to jail :)































































































































