import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

# Load data. Change winner to binary 1/0:
load( '../out/d3-fight-level-transform.pkl' )
print( X.shape )

# train test split.
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(
    X, y, 
    random_state = 718,
    test_size = 0.25
)

# Fit decision tree.
from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier( 
    n_estimators = 400, 
    max_depth = 15,
    min_samples_leaf = .01, 
    n_jobs = -1,
    random_state = 841
)
m.fit( X_train, y_train )

# accuracy against train data (in-model).
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
print( classification_report( 
    y_train, 
    m.predict(X_train)
))

# accuracy against test data.
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
print( classification_report( 
    y_test, 
    m.predict(X_test)
))
y_test.mean()

# check cross-validation.
crossval = False
if crossval: 
    from sklearn.model_selection import cross_validate
    kfold = cross_validate( 
        m, X_train, y_train, 
        cv = 10, 
        scoring = [ 'precision', 'recall' ], 
        n_jobs = -1 
    )
    print( 
      'Mean precision: %s \nMean recall %s' % 
      ( kfold['test_precision'].mean(), kfold['test_recall'].mean() ) 
    )
