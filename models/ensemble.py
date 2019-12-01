# read functions.
import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

# Load data.
load( '../out/d3-fight-level-transform.pkl' )

import random
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier

# train test split.
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(
    X, y, 
    random_state = 718,
    test_size = 0.25
)

# Fit
m = VotingClassifier( estimators = [
    ('Logistic Regression', LogisticRegression( solver = 'lbfgs', n_jobs = -1 )),
    ('ADA Booster', AdaBoostClassifier()),
    ('Decision Tree', DecisionTreeClassifier( 
        max_depth = 20, 
        min_samples_split = .03
    )),
    ('Random Forest', RandomForestClassifier( 
        n_estimators = 200, 
        min_samples_leaf = .1,
        n_jobs = -1
    )),
    ('SVM Linear', SVC( kernel='linear' )),
    ('SVM Gaussian-RBF', SVC( kernel='rbf' )) 
])
m.fit( X_train, y_train )

# accuracy against train data (in-model).
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
print( classification_report( y_train, m.predict(X_train) ) )

# accuracy against test data.
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
print( classification_report( y_test, m.predict(X_test) ) )

