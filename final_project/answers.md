# Enron Project

## 1. Project overview

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
 
 The dataset includes 20 features and 146 people (data points), of which 18 are 
 indentified as POI.

### 1.3 Outliers

I had a look on NaNs in the data. There was one person, who has only NaNs
("LOCKHART EUGENE E"). I googled him, but didn't find anything related to 
Enron, so I removed him from the data.

Scrolling through names I also saw "THE TRAVEL AGENCY IN THE PARK" and 
"TOTAL", which most likely aren't real people, so I removed them, too.


# 2. Features

### 2.1 Additional feature

I noticed, that number of NaNs varies from person to person, so I created a 
new feature, which counts number of Nans. I may be, that NaN, which in this 
case is most likely 0, mean that no payments were made. So, if person has 
many features, where payments was 0, the probability, that the person was 
POI, may be lower.

I also created two features `'ratio_to_poi'` and `'ratio_from_poi'`, which give 
the fraction of all emails sent or received divided by all emails sent / 
received. It might be, that some people got or sent a lot of emails every day
 and this features might say how many % of these emails were actually 
 contacting POIs. My hypothesis would be, that the higher percentual 
 frequency of contacting POIs, the higher chance this person is POI.

Last feature I created was `bonus_to_salary` what is fraction of bonus to 
salary. As another approach, I wanted to check, if POIs got much higher 
bonuses than their salaries. I would assume that in a big company bonus is 
either same for all (what would be actually then not changing anything as it 
would grow same as salary) or some % of the earnings. If this ratio bonus to 
salary would be extremally high, I would assume, that this person was a POI.

Using all my features improved the performance of algorithm in comparison to 
"default" features. But in my final features set, I ended up using only 
`ratio_to_poi` as it had one of the highest feature importances. As my final 
features set included one feature `'ratio_to_poi'`, I also tested algorithms 
 performace without this feature, leaving all other chosen features. I named
  this set `features_4.` It turned out, that the performance was a bit worse
  . For results comparison, please check Table in 2.2 Feature selection.

### 2.2 Feature selection

I decided to remove email adress, as it might work the same way as the 
signature in Susan emails - it has first name and surname and most likely these
 are not data we would use to predict something where labels are also first 
 name and surname.
 
Regarding feature selection, I was testing different methods. I used 
`SelectKBest` with different k. It enhanced the performance (of the algorithm,
 but it was not high enough.

```
k:  1 	Accuracy mean:  0.82611627907 	Precision mean:  0.213 	Recall mean:  0.188
k:  2 	Accuracy mean:  0.815406976744 	Precision mean:  0.203 	Recall mean:  0.203
k:  3 	Accuracy mean:  0.814046511628 	Precision mean:  0.217 	Recall mean:  0.229
k:  4 	Accuracy mean:  0.814069767442 	Precision mean:  0.224 	Recall mean:  0.241
k:  5 	Accuracy mean:  0.814502325581 	Precision mean:  0.230 	Recall mean:  0.251
k:  6 	Accuracy mean:  0.815189922481 	Precision mean:  0.236 	Recall mean:  0.259
k:  7 	Accuracy mean:  0.815790697674 	Precision mean:  0.240 	Recall mean:  0.263
k:  8 	Accuracy mean:  0.816485465116 	Precision mean:  0.243 	Recall mean:  0.268
k:  9 	Accuracy mean:  0.817012919897 	Precision mean:  0.246 	Recall mean:  0.272
k:  10 	Accuracy mean:  0.817606976744 	Precision mean:  0.248 	Recall mean:  0.275
k:  11 	Accuracy mean:  0.818071881607 	Precision mean:  0.250 	Recall mean:  0.277
k:  12 	Accuracy mean:  0.818360465116 	Precision mean:  0.251 	Recall mean:  0.278
k:  13 	Accuracy mean:  0.818745974955 	Precision mean:  0.253 	Recall mean:  0.280
k:  14 	Accuracy mean:  0.819071428571 	Precision mean:  0.254 	Recall mean:  0.283
k:  15 	Accuracy mean:  0.819271317829 	Precision mean:  0.255 	Recall mean:  0.284
k:  16 	Accuracy mean:  0.819542151163 	Precision mean:  0.256 	Recall mean:  0.286
k:  17 	Accuracy mean:  0.819699042408 	Precision mean:  0.256 	Recall mean:  0.287
k:  18 	Accuracy mean:  0.819826873385 	Precision mean:  0.257 	Recall mean:  0.289
k:  19 	Accuracy mean:  0.819958384333 	Precision mean:  0.258 	Recall mean:  0.290
k:  20 	Accuracy mean:  0.819977906977 	Precision mean:  0.258 	Recall mean:  0.291
k:  21 	Accuracy mean:  0.82 	        Precision mean:  0.258 	Recall mean:  0.291
k:  22 	Accuracy mean:  0.81989217759 	Precision mean:  0.258 	Recall mean:  0.292
```

So I decided to choose features manually. First, I ran algorithm 1000 times 
with all feature but my added and check accuracy, presision and recall. Then,
 I did the same, but with my added features. This already improved accuracy,
  but not enough.

I decided to perform with removing features, which have importance 0, in 
order to use only most important features. First, I printed all mean feature 
importances from 1000 runs using all features, including mine.

```
Mean importances: 
salary :  0.023
deferral_payments :  0.006
total_payments :  0.033
loan_advances :  0.000
bonus :  0.099
restricted_stock_deferred :  0.001
deferred_income :  0.050
total_stock_value :  0.058
expenses :  0.097
exercised_stock_options :  0.122
other :  0.083
long_term_incentive :  0.033
restricted_stock :  0.033
director_fees :  0.000
to_messages :  0.015
from_poi_to_this_person :  0.023
from_this_person_to_poi :  0.025
shared_receipt_with_poi :  0.067
number_info :  0.023
bonus_to_salary :  0.047
ratio_to_poi :  0.146
ratio_from_poi :  0.017
```

We can see that `loan_advances` and `director_fees` have importance 0.0. I'll
 remove them and check it the recall and precision will be better. If 
 performance and recall got better, I will keep on removing features with 
 lowest importances (not only 0), till some step, when recall and performance
  will get worse. From this I got a following set of features, which I will 
  call features_1 in the table: `features_list = ['poi', 'bonus', 'expenses',
   'exercised_stock_options', 'other', 'shared_receipt_with_poi', 'ratio_to_poi']`.
                 
As basically same approach, but done from other side, I decided to 
run the algorithm one more time and choose 5 (features_2) and 7 (features_4) 
features with highest feature importances (which is one more and one less 
than in previous method and both of them should give less precision and recall.

features_2: `features_list = ['poi', 'bonus', 'expenses', 'exercised_stock_options', 
                 'other', 'ratio_to_poi']`

features_3: `features_list = ['poi', 'bonus', 'total_stock_value', 
'expenses', 'exercised_stock_options', 'other', 'shared_receipt_with_poi', 'ratio_to_poi']`

features_4: `features_list = ['poi', 'bonus', 'expenses', 'exercised_stock_options',
                 'other', 'shared_receipt_with_poi']`


I tried to scale my best feature combination (features_1), but it worse the 
performance a bit.

| Features        | Precision           | Recall  |
| :-----------: |:-------------------:| :------:|
| all default features, no scaling | 0.222273015873 | 0.2546 |
| all features, included mine, no scaling | 0.250516225441 | 0.2964 |
| SelectKBest(k=22) | 0.258 | 0.292 |
| features_1 | 0.361069053169 | 0.3834 |
| features_2 | 0.335161824287 | 0.3584 |
|features_3| 0.354247757798 | 0.3742 |
|features_4|0.293598088023|0.3208|
| features_1 + MinMaxScaler() | 0.358376875902 | 0.384 |


After chosing features, I tried PCA for all algorithms, but here also, it 
didn't improve performance of all algorithms.

In my final algorithm, I chosen features_1: `features_list = ['poi', 'bonus',
 'expenses', 'exercised_stock_options', 'other', 'shared_receipt_with_poi', 
 'ratio_to_poi']` without scaling nor PCA.

## 3. Algorithm

I tried decision trees, random forest, SVM and AdaBoost.

### 3.1 Random Forest

Testing random forest, I used SelectKBest with different parameters. After 
it, I tried PCA, but it didn't optimize performance of the algorithm. Lastly,
 I added feature scaling with MinMaxScaler. However, the performance of this 
 algorithm was not enough.
  
### 3.2 Decision Trees
  
Then I tried decision trees. I described my feature selection as my final 
feature set in point 2.2. After selecting features, I was testing best 
minimal samples split, and the best result was by `min_samples_split=2`. 
Then, I tried to use PCA with different components 
 number. What I found interesting, the highest performance was achieved when
  PCA was set up to 1... but after validation it wasn't high enough. So, I 
  decided to not use PCA. I tried also scaling, but it didn't enhance 
  algorithm's performance.

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
|Decision Trees|0.361069053169 | 0.3834 |
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

Tuning parameters should be performed carefully. If parameters are 
under-tuned, we don't use all performance of algorithm, we could use. It 
means, tuning up parameters in this case will most probably enhance recall or
 precision or both of these. Opposite situation is over-tuning. If parameters
  are over-tuned, algorithm works very good on training set, but it has bad 
  performance on test set and real data. The border is not easy to 
  find as it might be a fine line between over-tuning and under-tuning.
 
## 5. Validation

Validation is a metric, which tells if the model will be also valid when used
 on general data (which is not test neither training data). Usually, already 
 too high accuracy (somewhere near 1), will tell us, that we are overfitting 
 the classifier and it will be useless with another data. In the Enron 
 example, it is a bit difficult, because we have very less data. The more 
 data we have, the more training and test cases we can generate, the more 
 sure we can be about validating the model.
 
 In order to validate the algorithm in best possible way, test set should be 
 as big as possible. With huge datasets it is ok to have 50/50 test/training 
 data or 0.33 for test/training/validation. But this is not our case. 0.5 may
  be too less training data, so I decided to take 0.7 training and 0.3 test 
  data. Keeping this balance in splitting the data is stratification. I use 
  randomized stratification method - StratifiedShuffleSplit().
  
  Apart of choosing the split, multiple combinations has to be tested in 
  order to not base our decision on a one-off good data split. This part is 
  called 1000-fold cross-validation and what I mean by it, I take 1000 
  stratified shuffle splits and every single time I fit the algorithm and 
  look at it's performance. Then, I calculate mean of these 1000 performances
   and use it to validate my model.
 
 ## 6. Performance metrics
 
 To measure my performance I used precision and recall. I had an eye on 
 accuracy, but it wasn't important metric for me.
 
 Precision tells us, how many people we identified correctly. It means, how 
 many people we identified correctly as POIs out of all people we identified 
 as POI. The higher precision is, the less people are wrongly identified as 
 non POI, when they are POI in real life.
   
 Recall tells us also how many people we identified correctly. In opposition 
 to performance, recall shows, how many people are not identified incorrectly
 . The higher recall is, the more sure we are, that none people are 
 identified incorrectly. It means, that if recall equals 1, we are 
 sure, that all identified people can go straight to jail :)































































































































