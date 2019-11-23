# read functions.
import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

# Load data. Change winner to binary 1/0:
load( '../out/d3-fight-level-standardize-normalize.pkl' )
y[ y == -1 ] = 0

# train test split.
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(
    X, y, 
    random_state = 718,
    test_size = 0.25
)

# Fit decision tree.
from sklearn.tree import DecisionTreeClassifier
m = DecisionTreeClassifier( max_depth = 14, min_samples_split = .05, random_state = 716 )
m.fit( X_train, y_train )

# accuracy against train data (in-model).
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
print( classification_report( y_train, m.predict(X_train) ) )

# accuracy against test data.
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
print( classification_report( y_test, m.predict(X_test) ) )

