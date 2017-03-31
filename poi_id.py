from __future__ import division
import sys
import numpy as np
import pickle
import pandas as pd 
sys.path.append("../../tools/")
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Load the dictionary containing the dataset
enron_data = pickle.load(open("final_project_dataset.pkl", "r"))
enron_df = pd.DataFrame(enron_data)
###Transopse the above dataframe we get
enron_tr = enron_df.transpose()
### How many datapints 

print 'The number of data point in the enron dataset is:', len(enron_tr)
print 'The number of features is :', len(enron_tr.columns)


list1 = enron_tr.columns
def count_nan(x,l):
    for i in list1:
        print x[i].value_counts(dropna=False)
count_nan(enron_tr,list1)
list1 = enron_tr.columns
list2 = [64,107,97,129,35,44,51,60,60,60,142,80,53,0,36,128,51,60,60,21,20]
dict1 = {'Feature':list1,'Number of NaNs':list2}
missing_df = pd.DataFrame(dict1)
missing_df
#### Proportion  of poi's and non-pois

coln = enron_df.columns
no_poi = []
for i in coln:
    no_poi.append((enron_df[i]['poi']!=False))
tot_poi = sum(no_poi)
tot_non_poi = len(no_poi)-sum(no_poi)       
print 'The percentage poi is:', (tot_poi/len(enron_tr))*100,"%"
print "And the percentage of non-poi is:",(tot_non_poi/len(enron_tr))*100,"%"

### Load the dictionary containing the dataset
import matplotlib.pyplot as plt


    
### Task 2: Remove outliers
### To examine outlier let us scatter plot some of the features
plt.scatter( enron_tr['salary'], enron_tr['bonus'] )
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show()
### The 'TOTAL' row of the dataset have outlier for bonus and salary and should be
### removed as below
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict.pop("TOTAL")
my_dataset = data_dict
my_df = pd.DataFrame(data_dict)
my_df_tr = my_df.transpose()
plt.scatter( my_df_tr['salary'], my_df_tr['bonus'] )
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show()

### Task 3: Create new feature(s)
### The script below will create two features: fraction of all messages to this person that come from POIs
###  and
###  fraction of all messages from this person that are sent to POIs

for name in my_dataset:
    data_point = data_dict[name]
    if (all([data_point['from_poi_to_this_person'] != 'NaN',data_point['from_this_person_to_poi'] != 'NaN',data_point['to_messages'] != 'NaN',data_point['from_messages'] != 'NaN'])):
        fraction_from_poi = float(data_point["from_poi_to_this_person"]) / float(data_point["to_messages"])
        data_point["fraction_from_poi"] = fraction_from_poi
        fraction_to_poi = float(data_point["from_this_person_to_poi"]) / float(data_point["from_messages"])
        data_point["fraction_to_poi"] = fraction_to_poi
    else:
        data_point["fraction_from_poi"] = data_point["fraction_to_poi"] = 0    
### Extract features and labels from dataset for local testing
from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler
feature_list = ['poi','salary', 'bonus','fraction_from_poi', 'fraction_to_poi','deferral_payments', 
                             'deferred_income','exercised_stock_options', 'expenses','loan_advances', 
                                'long_term_incentive', 'restricted_stock', 'restricted_stock_deferred', 
                             'shared_receipt_with_poi', 'total_payments', 'total_stock_value','director_fees','other',
                'from_this_person_to_poi','from_poi_to_this_person','to_messages','from_messages']
data = featureFormat(data_dict, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


###Scaling feature 

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Feature selection using SelectKBest

from sklearn.feature_selection import *
k_top_features = SelectKBest(k=21) #score_func=f_regression,
k_top_features.fit(features,labels)
top_list = zip(k_top_features.get_support(), feature_list[1:], k_top_features.scores_)
top_list = sorted(top_list, key=lambda x: x[2], reverse=True)
print "The top list of features:", pd.DataFrame(top_list)

### list of selected features after MinMaxScaler
feature_list =  ['poi','exercised_stock_options','total_stock_value','bonus','salary',
                 'fraction_to_poi','deferred_income','long_term_incentive']
### Train/test split using train_test_split of sklearn
from sklearn.cross_validation import train_test_split
data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.3, random_state=42)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.calibration import calibration_curve
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


#param_grid = {
#     'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#         'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
#         }

#svc = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)    
#gnb = GaussianNB()
#lgr = LogisticRegression()
#svc = SVC()
#svcl = LinearSVC()
#rfc = RandomForestClassifier(n_estimators=100)
#dct = tree.DecisionTreeClassifier()
#adb = AdaBoostClassifier()
#sgd = SGDClassifier()
#for clf,name in [(gnb, 'Naive Bayes'),
#                (sgd, 'Stochastic gradient decent '),
#                (dct, 'Decition Tree classification'),
#                (adb, 'AdaBoost Classifier'),
#                (svc, 'Support Machine Vector'),
#                (lgr, 'Logistic Regression'),
#                (svcl, 'Linear SVC'), 
#                (rfc, 'Random Forest') ]:
#    t0 = time()
#    clf.fit(features_train, labels_train)
#    prediction= clf.predict(features_test)
#    print 'Time to complete the model run:', round(time()-t0,4),'s'
#    score=metrics.accuracy_score(labels_test, prediction)
#    class_report =classification_report(labels_test, prediction)
#    print 'The score for classification algorithm is:', name, score
#    print  class_report, name
### Tuning all the above algoriths for better performance using the test_classifier 
### from  test.py  script and compare the score with  scores from the above procedure 
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from tester import dump_classifier_and_data, test_classifier
clf = GaussianNB()
t0 = time()
test_classifier(clf, my_dataset, feature_list, folds = 1000)
print 'Time to complete the model run:', round(time()-t0,4),'s'
### Task 6: D
##dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, feature_list)
